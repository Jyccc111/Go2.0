"""Bounded, repeatable probes for diagnosing value training/evaluation gaps."""

import json
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ValueDiagnostics:
    def __init__(self, train_files, val_files, seed, chunks, samples, report_path):
        if chunks < 1 or samples < 1:
            raise ValueError("diagnostic_chunks and diagnostic_samples must be positive")
        self.samples = samples
        self.report_path = report_path
        self.run_id = datetime.now(timezone.utc).isoformat()
        self.rng = np.random.default_rng(seed)
        self.selected = {}
        self.parts = {"train": [], "val": []}
        self.probes = {}
        for split, files in (("train", train_files), ("val", val_files)):
            indices = self.rng.choice(len(files), min(chunks, len(files)), replace=False)
            self.selected[split] = {files[int(i)] for i in indices}
        self.record({
            "event": "split", "seed": seed,
            "train_files": train_files, "val_files": val_files,
            "probe_files": {key: sorted(value) for key, value in self.selected.items()},
            "samples_per_probe_file": samples,
        })
        print(f"诊断报告追加写入 {report_path}，run_id={self.run_id}")
        print("诊断固定抽样局面，不能替代独立棋谱测试，也不能证明标签语义正确。")

    def record(self, record):
        with open(self.report_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(
                {"run_id": self.run_id, **record}, ensure_ascii=False, allow_nan=False
            ) + "\n")

    def capture(self, split, filename, states, results):
        """Called on the first epoch while each chunk is already in memory."""
        labels = np.asarray(results).reshape(-1)
        black = int(np.count_nonzero(labels == 1))
        print(f"[Data {split}] N={len(labels)}, 黑胜={black / len(labels):.2%}, "
              f"states dtype={states.dtype}")
        self.record({
            "event": "chunk", "split": split, "file": filename,
            "shape": list(states.shape), "dtype": str(states.dtype),
            "white": int(len(labels) - black), "black": black,
        })
        if filename not in self.selected[split]:
            return
        indices = self.rng.choice(len(labels), min(self.samples, len(labels)), replace=False)
        sampled = np.asarray(states[indices], dtype=np.float32)
        if not np.isfinite(sampled).all():
            raise ValueError(f"Non-finite sampled input in {filename}")
        axes = (0, 2, 3)
        stats = {
            "min": sampled.min(axis=axes).tolist(),
            "max": sampled.max(axis=axes).tolist(),
            "mean": sampled.mean(axis=axes).tolist(),
            "std": sampled.std(axis=axes).tolist(),
        }
        binary_planes = sampled[:, [0, 1, 7]]
        checks = {
            "invalid_binary_cells": int(np.count_nonzero(
                (binary_planes != 0) & (binary_planes != 1))),
            "overlapping_stones": int(np.count_nonzero(
                (sampled[:, 0] != 0) & (sampled[:, 1] != 0))),
            "legal_occupied_points": int(np.count_nonzero(
                (sampled[:, 7] != 0) & ((sampled[:, 0] != 0) | (sampled[:, 1] != 0)))),
            "negative_feature_values": int(np.count_nonzero(sampled < 0)),
        }
        print(f"[Features {split}] {filename}\n"
              f"  各通道 min={stats['min']}\n  各通道 max={stats['max']}")
        if any(checks.values()):
            print(f"  输入检查发现异常 {checks}")
        self.record({"event": "features", "split": split, "file": filename,
                     "checks": checks, **stats})
        self.parts[split].append((sampled, labels[indices].astype(np.float32)))

    def run(self, model, epoch, batch_size):
        if not self.probes:
            for split, parts in self.parts.items():
                states = np.concatenate([part[0] for part in parts])
                labels = np.concatenate([part[1] for part in parts])
                # Mix positions from the selected chunks for the BN comparison.
                order = self.rng.permutation(len(labels))
                self.probes[split] = (
                    torch.from_numpy(states[order]), torch.from_numpy(labels[order])
                )
            self.parts.clear()

        metrics = {}
        for split, (states, labels) in self.probes.items():
            for batch_stats in (False, True):
                name = f"{split}_{'batch_bn' if batch_stats else 'eval'}"
                values = self.evaluate(model, states, labels, batch_size, batch_stats)
                metrics[name] = values
                print(f"[Probe {name}] N={len(labels)}, loss={values['loss']:.4f}, "
                      f"accuracy={values['accuracy']:.2%}, "
                      f"高置信错误占比={values['confident_wrong_rate']:.2%}")
                print(f"  confusion [White, Black]={values['confusion_matrix']}")
                print(f"  真实黑胜比例={values['black_label_rate']:.2%}, "
                      f"平均预测黑胜概率={values['mean_black_probability']:.2%}")
        self.record({"event": "epoch", "epoch": epoch, "metrics": metrics})
        return metrics

    @staticmethod
    def evaluate(model, states, labels, batch_size, batch_stats=False):
        """Probe with frozen weights; restore BN buffers and every module's mode."""
        modes = {module: module.training for module in model.modules()}
        bn_layers = [module for module in model.modules()
                     if isinstance(module, nn.modules.batchnorm._BatchNorm)]
        buffers = [(buffer, buffer.detach().clone())
                   for module in bn_layers for buffer in module.buffers()]
        outputs = []
        device = next(model.parameters()).device
        try:
            model.eval()
            if batch_stats:
                for module in bn_layers:
                    module.train()
            with torch.no_grad():
                for start in range(0, len(labels), batch_size):
                    output = model(states[start:start + batch_size].to(device))
                    if output.shape != (len(states[start:start + batch_size]), 1):
                        raise ValueError(f"Unexpected value output shape {tuple(output.shape)}")
                    outputs.append(output.flatten().cpu())
        finally:
            with torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)
            for module, training in modes.items():
                module.training = training

        logits = torch.cat(outputs)
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite value logits in diagnostic probe")
        probabilities = logits.sigmoid()
        predicted = (logits >= 0).long()
        actual = labels.long()
        confusion = torch.bincount(2 * actual + predicted, minlength=4).reshape(2, 2)
        wrong = predicted != actual
        confident = (probabilities <= 0.1) | (probabilities >= 0.9)
        black_rate = labels.mean().item()
        return {
            "loss": F.binary_cross_entropy_with_logits(logits, labels).item(),
            "accuracy": (predicted == actual).float().mean().item(),
            "confusion_matrix": confusion.tolist(),
            "black_label_rate": black_rate,
            "majority_accuracy": max(black_rate, 1 - black_rate),
            "mean_black_probability": probabilities.mean().item(),
            "brier_score": ((probabilities - labels) ** 2).mean().item(),
            "confident_wrong_rate": (wrong & confident).float().mean().item(),
            "flipped_label_loss": F.binary_cross_entropy_with_logits(logits, 1 - labels).item(),
        }
