import gc
import os
from value_diagnostics import ValueDiagnostics
from huggingface_hub import HfApi
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch.nn.functional as F

def get_device():
    """Prefer CUDA, then Apple MPS, and fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()


def validate_value_arrays(states, results):
    labels = np.asarray(results).reshape(-1)
    if states.ndim != 4 or tuple(states.shape[1:]) != (8, 19, 19):
        raise ValueError(f"states must have shape (N, 8, 19, 19); got {states.shape}")
    if len(states) != len(labels) or not len(labels):
        raise ValueError("states and results must have the same non-zero length")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("results must be binary labels: 1 for Black win, 0 for White win")


def build_mixed_pool(files, samples_per_chunk, max_pool_samples, seed, diagnostic=None):
    """Sample each training chunk once, then reuse and shuffle the bounded pool."""
    if max_pool_samples < len(files):
        raise ValueError("max_pool_samples must allow at least one sample per training chunk")
    rng = np.random.default_rng(seed)
    quota, remainder = divmod(max_pool_samples, len(files))
    state_parts = []
    label_parts = []
    for index, filename in enumerate(sorted(files)):
        print(f"[Pool {index + 1}/{len(files)}] {filename}")
        with np.load(filename, allow_pickle=False) as data:
            states = data["states"]
            labels = data["results"].reshape(-1)
        validate_value_arrays(states, labels)
        count = min(len(labels), samples_per_chunk, quota + (index < remainder))
        indices = rng.choice(len(labels), count, replace=False)
        sampled_states = states[indices]
        sampled_labels = labels[indices]
        if not np.isfinite(sampled_states).all():
            raise ValueError(f"Non-finite input in sampled pool from {filename}")
        state_parts.append(sampled_states)
        label_parts.append(sampled_labels)
        if diagnostic is not None:
            diagnostic.record({
                "event": "pool_selection", "file": filename,
                "source_samples": len(labels), "pool_samples": count,
                "source_black": int(np.count_nonzero(labels == 1)),
                "source_indices": indices.tolist(),
            })
            # The training probe must contain positions actually used for updates.
            diagnostic.capture("train", filename, sampled_states, sampled_labels)
        del states, labels
    dataset = GoDataset(np.concatenate(state_parts), np.concatenate(label_parts))
    print(f"Mixed pool: {len(dataset):,} positions from {len(files)} training chunks")
    return dataset


class GoDataset(Dataset):

    def __init__(self, states, results):

        self.states = torch.tensor(
            states,
            dtype=torch.float32
        )

        self.results = torch.as_tensor(results, dtype=torch.float32).reshape(-1)

        if self.states.ndim != 4 or tuple(self.states.shape[1:]) != (8, 19, 19):
            raise ValueError(
                f"states must have shape (N, 8, 19, 19); got {tuple(self.states.shape)}"
            )
        if len(self.states) != len(self.results):
            raise ValueError("states and results must contain the same number of samples")
        if len(self.results) == 0:
            raise ValueError("dataset contains no samples")
        if not torch.all((self.results == 0) | (self.results == 1)):
            raise ValueError("results must be binary labels: 1 for Black win, 0 for White win")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):

        return (
            self.states[index],
            self.results[index]
        )
class ResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Residual connection
        x = x + identity

        x = F.relu(x)

        return x
class Value_network(nn.Module):

    def __init__(self, size=64, num_blocks=4):

        super().__init__()

        if isinstance(size, bool) or int(size) != size or size <= 0:
            raise ValueError("size must be a positive integer")
        if isinstance(num_blocks, bool) or int(num_blocks) != num_blocks or num_blocks < 0:
            raise ValueError("num_blocks must be a non-negative integer")
        self.size = int(size)
        self.num_blocks = int(num_blocks)

        self.network = nn.Sequential(

            nn.Conv2d(
                in_channels=8,
                out_channels=self.size,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(self.size),
            nn.ReLU(),

            # ResNet
            *[ResBlock(self.size) for _ in range(self.num_blocks)],
            nn.Conv2d(
                size,
                2,
                kernel_size=1
            ),
            nn.BatchNorm2d(2),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(
                2 * 19 * 19,
                1
            ),

        )
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
    def forward(self, x):
        return self.network(x)

    def load(self, filename):
        checkpoint = torch.load(
        filename,
        map_location=device
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if state_dict and next(iter(state_dict)).startswith("network."):
            state_dict = {key.removeprefix("network."): value for key, value in state_dict.items()}
        self.network.load_state_dict(state_dict)

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        return checkpoint.get("epoch", 0)

    def fit(
            self,
            epochs,
            files,
            data_repo_id,
            model_repo_id,
            val_ratio=0.2,
            batch_size=512,
            diagnostics=True,
            diagnostic_chunks=3,
            diagnostic_samples=1024,
            diagnostic_report="value_diagnostics.jsonl",
            split_seed=42,
            upload_models=True,
            training_mode="mixed",
            samples_per_chunk=4096,
            max_pool_samples=65536,
            trace_chunk_probes=True
        ):
        """Train on a fixed, shuffled cross-chunk pool by default.

        samples_per_chunk and max_pool_samples bound the mixed training pool.
        Validation still uses all validation positions. Use training_mode='chunk'
        to run the original sequential training with optional retention probes.
        """

        api = HfApi() if upload_models else None
        self.to(device)

        if epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not files:
            raise ValueError("files must contain at least two data chunks")
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if len(files) < 2:
            raise ValueError("at least two data chunks are required for train/validation split")
        if training_mode not in ("mixed", "chunk"):
            raise ValueError("training_mode must be 'mixed' or 'chunk'")
        for name, value in (("samples_per_chunk", samples_per_chunk),
                            ("max_pool_samples", max_pool_samples)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        # The data uploader stores 1 for a Black win and 0 for a White win.
        # The network emits one logit, so BCEWithLogitsLoss is the matching,
        # numerically stable objective.
        criterion = nn.BCEWithLogitsLoss()

        # ==========================================
        # 划分 Train / Validation
        # ==========================================

        files = sorted(os.path.realpath(filename) for filename in files)
        if len(files) != len(set(files)):
            raise ValueError("files contains duplicate chunk paths")
        split_rng = random.Random(split_seed)
        split_rng.shuffle(files)

        split = int(len(files) * (1 - val_ratio))

        train_files = files[:split]
        val_files = files[split:]

        if not train_files or not val_files:
            raise ValueError("the selected val_ratio produced an empty train or validation split")

        print(
            f"Train chunks: {len(train_files)}"
        )

        print(
            f"Validation chunks: {len(val_files)}"
        )

        print(f"Split seed: {split_seed}")
        print("Train files:", train_files)
        print("Validation files:", val_files)
        diagnostic = None
        if diagnostics:
            diagnostic = ValueDiagnostics(
                train_files, val_files, split_seed, diagnostic_chunks,
                diagnostic_samples, diagnostic_report,
            )
            diagnostic.record({
                "event": "training_config", "training_mode": training_mode,
                "samples_per_chunk": samples_per_chunk,
                "max_pool_samples": max_pool_samples,
                "trace_chunk_probes": trace_chunk_probes,
            })

        mixed_dataset = None
        if training_mode == "mixed":
            mixed_dataset = build_mixed_pool(
                train_files, samples_per_chunk, max_pool_samples, split_seed, diagnostic,
            )

        # ==========================================
        # Learning Rate Scheduler
        # ==========================================

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )

        # ==========================================
        # Early Stopping
        # ==========================================

        best_val_loss = float("inf")

        patience = 5
        bad_epochs = 0

        # ==========================================
        # Epoch
        # ==========================================

        for epoch in range(epochs):

            print()
            print(
                f"========== Epoch {epoch + 1} =========="
            )

            # ======================================
            # TRAIN
            # ======================================

            self.train()

            split_rng.shuffle(train_files)

            train_loss = 0
            train_samples = 0
            epoch_confusion_matrix = torch.zeros(
                (2, 2),
                dtype=torch.long,
                device=device,
            )

            if training_mode == "mixed":
                training_groups = [("mixed_pool", mixed_dataset)]
            else:
                training_groups = [(filename, None) for filename in train_files]
            if diagnostic is not None:
                diagnostic.record({
                    "event": "training_order", "epoch": epoch + 1,
                    "groups": [filename for filename, _ in training_groups],
                })

            for i, (filename, prepared_dataset) in enumerate(training_groups):
                train_confusion_matrix = torch.zeros(
                                (2, 2),
                                dtype=torch.long,
                                device=device,
                            )
                print(
                    f"[Train {i + 1}/{len(training_groups)}] "
                    f"训练数据: {filename}"
                )

                # -------------------------------
                # Load
                # -------------------------------

                dataset = prepared_dataset
                if dataset is None:
                    with np.load(filename, allow_pickle=False) as data:
                        states = data["states"]
                        results = data["results"]
                    dataset = GoDataset(states, results)
                    if diagnostic is not None and epoch == 0:
                        diagnostic.capture("train", filename, states, results)
                    del states, results

                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True
                )

                # -------------------------------
                # Training
                # -------------------------------

                chunk_loss = 0.0
                chunk_samples = 0

                for states_batch, results_batch in loader:
                    states_batch = states_batch.to(device)
                    results_batch = results_batch.to(device).unsqueeze(1)

                    self.optimizer.zero_grad()

                    output = self(states_batch)

                    # Count this batch's predictions before updating weights.
                    predicted = (output >= 0).long()
                    actual = results_batch.long()
                    train_confusion_matrix[0, 0] += (
                        (actual == 0) & (predicted == 0)
                    ).sum()
                    train_confusion_matrix[0, 1] += (
                        (actual == 0) & (predicted == 1)
                    ).sum()
                    train_confusion_matrix[1, 0] += (
                        (actual == 1) & (predicted == 0)
                    ).sum()
                    train_confusion_matrix[1, 1] += (
                        (actual == 1) & (predicted == 1)
                    ).sum()

                    loss = criterion(
                        output,
                        results_batch
                    )

                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite training loss in {filename}")
                    loss.backward()

                    self.optimizer.step()

                    batch_size_actual = states_batch.size(0)

                    train_loss += (
                            loss.item()
                            * batch_size_actual
                    )

                    train_samples += batch_size_actual
                    chunk_loss += loss.item() * batch_size_actual
                    chunk_samples += batch_size_actual

                print(
                    f"Training group average loss: {chunk_loss / chunk_samples:.4f}"
                    if chunk_samples else "Chunk average loss: n/a (empty chunk)"
                )
                print(
                    "Online train confusion matrix "
                    "(rows=true [White, Black], columns=predicted [White, Black]):"
                )
                print(train_confusion_matrix.cpu().tolist())
                epoch_confusion_matrix += train_confusion_matrix
                if diagnostic is not None and training_mode == "chunk" and trace_chunk_probes:
                    diagnostic.trace_training(self, epoch + 1, filename, batch_size)
                # -------------------------------
                # Free memory
                # -------------------------------

                del dataset
                del loader

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # ======================================
            # Average Train Loss
            # ======================================

            if train_samples == 0:
                raise ValueError("training split contains no samples")
            train_loss /= train_samples
            

            # ======================================
            # VALIDATION
            # ======================================

            self.eval()

            val_loss = 0
            val_samples = 0
            confusion_matrix = torch.zeros(
                (2, 2),
                dtype=torch.long,
                device=device,
            )

            with torch.no_grad():

                for i, filename in enumerate(val_files):

                    print(
                        f"[Val {i + 1}/{len(val_files)}] "
                        f"正在读取: {filename}"
                    )

                    with np.load(filename, allow_pickle=False) as data:
                        states = data["states"]
                        results = data["results"]

                    dataset = GoDataset(
                        states,
                        results
                    )

                    if diagnostic is not None and epoch == 0:
                        diagnostic.capture("val", filename, states, results)
                    chunk_val_loss = 0.0
                    chunk_val_correct = 0
                    chunk_val_samples = 0

                    loader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False
                    )

                    for states_batch, results_batch in loader:
                        states_batch = states_batch.to(device)
                        results_batch = results_batch.to(device).unsqueeze(1)

                        output = self(states_batch)

                        # BCEWithLogitsLoss receives raw logits. A logit of 0
                        # corresponds to a predicted probability of 0.5.
                        predicted = (output >= 0).long()
                        actual = results_batch.long()
                        confusion_matrix[0, 0] += (
                            (actual == 0) & (predicted == 0)
                        ).sum()
                        confusion_matrix[0, 1] += (
                            (actual == 0) & (predicted == 1)
                        ).sum()
                        confusion_matrix[1, 0] += (
                            (actual == 1) & (predicted == 0)
                        ).sum()
                        confusion_matrix[1, 1] += (
                            (actual == 1) & (predicted == 1)
                        ).sum()

                        loss = criterion(
                            output,
                            results_batch
                        )
                        

                        batch_size_actual = states_batch.size(0)

                        val_loss += (
                                loss.item()
                                * batch_size_actual
                        )

                        if not torch.isfinite(loss):
                            raise ValueError(f"Non-finite validation loss in {filename}")
                        val_samples += batch_size_actual
                        chunk_val_loss += loss.item() * batch_size_actual
                        chunk_val_correct += int((predicted == actual).sum().item())
                        chunk_val_samples += batch_size_actual

                    print(f"[Val chunk] loss={chunk_val_loss / chunk_val_samples:.4f}, "
                          f"accuracy={chunk_val_correct / chunk_val_samples:.2%}")
                    if diagnostic is not None:
                        diagnostic.record({
                            "event": "validation_chunk", "epoch": epoch + 1,
                            "file": filename, "samples": chunk_val_samples,
                            "loss": chunk_val_loss / chunk_val_samples,
                            "accuracy": chunk_val_correct / chunk_val_samples,
                        })

                    # -------------------------------
                    # Free memory
                    # -------------------------------

                    del data
                    del states
                    del results
                    del dataset
                    del loader

                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # ======================================
            # Average Validation Loss
            # ======================================

            if val_samples == 0:
                raise ValueError("validation split contains no samples")
            val_loss /= val_samples
            print(
                "Validation confusion matrix "
                "(rows=true [White, Black], columns=predicted [White, Black]):"
            )
            print(confusion_matrix.cpu().tolist())

            print(f"Online train accuracy: "
                  f"{epoch_confusion_matrix.diag().sum().item() / train_samples:.2%}")
            print(f"Validation accuracy: "
                  f"{confusion_matrix.diag().sum().item() / val_samples:.2%}")
            diagnostic_metrics = None
            if diagnostic is not None:
                diagnostic_metrics = diagnostic.run(self, epoch + 1, batch_size)
                diagnostic.record({
                    "event": "epoch_totals", "epoch": epoch + 1,
                    "training_samples": train_samples,
                    "online_train_loss": train_loss, "validation_loss": val_loss,
                    "online_train_confusion": epoch_confusion_matrix.cpu().tolist(),
                    "validation_confusion": confusion_matrix.cpu().tolist(),
                })

            # ======================================
            # Learning Rate
            # ======================================

            scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            print()
            print(
                f"Epoch {epoch + 1}"
            )

            print(
                f"Train Loss: {train_loss:.4f}"
            )

            print(
                f"Val Loss:   {val_loss:.4f}"
            )

            print(
                f"Learning Rate: {current_lr:.8f}"
            )

            # ======================================
            # Save Best Model + Last Model
            # ======================================

            # --------------------------------------
            # 判断是否是最佳模型
            # --------------------------------------

            is_best = val_loss < best_val_loss

            if is_best:

                best_val_loss = val_loss
                bad_epochs = 0

            else:

                bad_epochs += 1

                print(
                    f"Validation 没有改善 "
                    f"({bad_epochs}/{patience})"
                )

            # ======================================
            # 创建当前 Epoch checkpoint
            # ======================================

            checkpoint = {
                "training_samples": train_samples,
                "training_mode": training_mode,
                "samples_per_chunk": samples_per_chunk,
                "max_pool_samples": max_pool_samples,
                "train_files": sorted(train_files),
                "val_files": val_files,
                "split_seed": split_seed,
                "diagnostic_metrics": diagnostic_metrics,
                "size": self.size,
                "num_blocks": self.num_blocks,

                "epoch": epoch + 1,

                "model_state_dict":
                    self.network.state_dict(),

                "optimizer_state_dict":
                    self.optimizer.state_dict(),

                "best_val_loss":
                    best_val_loss,

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "learning_rate":
                    current_lr
            }

            # ======================================
            # 保存最新模型
            # 每个 Epoch 都保存
            # ======================================

            filename_last = "last_value_network.pth"

            torch.save(
                checkpoint,
                filename_last
            )

            if upload_models:
                api.upload_file(
                    path_or_fileobj=filename_last,
                    path_in_repo=filename_last,
                    repo_id=model_repo_id,
                    repo_type="model"
                )

            print(
                "✓ 已保存最新模型"
            )

            # ======================================
            # 如果是最佳模型
            # ======================================

            if is_best:
                filename_best = "best_value_network.pth"

                torch.save(
                    checkpoint,
                    filename_best
                )

                if upload_models:
                    api.upload_file(
                        path_or_fileobj=filename_best,
                        path_in_repo=filename_best,
                        repo_id=model_repo_id,
                        repo_type="model"
                    )

                print(
                    "★ 保存新的最佳模型"
                )

            # ======================================
            # Early Stopping
            # ======================================

            if bad_epochs >= patience:
                print()
                print(
                    "Early stopping!"
                )

                break
