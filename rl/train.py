"""Epoch-based REINFORCE: exact effective updates and one value sample per game."""
import argparse
import copy
import json
import hashlib
import shutil
import tempfile
import os
import zipfile
from datetime import datetime, timezone
import uuid
from pathlib import Path
import random
import torch
from .game import GoAdapter
from .policy import distribution, load_policy, save_checkpoint
from .hub import resolve_checkpoint, save_game, upload_training, snapshot_code, bootstrap_opponent

from .opponents import discover_history, download_version, HistoricalOpponentPool
from .value_data import ReservoirPositionSampler, EpochValueDataset, upload_epoch_value


def rollout(model, env, opponent=None, learner_color=1, temperature=1.0, sampler=None):
    model.eval()  # Freeze BatchNorm statistics; eval still allows gradients.
    if opponent is not None:
        opponent.eval()
    records = []
    device = next(model.parameters()).device
    while not env.done:
        color = env.player
        learner = opponent is None or color == learner_color
        actor = model if learner else opponent
        state, mask = env.observe(), env.legal_mask()
        if sampler is not None:
            sampler.consider(state, env.game.board.board, env.moves, color)
        with torch.no_grad():
            logits = actor(torch.as_tensor(state, dtype=torch.float32, device=device)[None])[0]
            action = int(distribution(logits, mask, temperature).sample().item())
        if learner:
            records.append((state, mask, action, color))
        env.step(action)
    return records


def reinforce_update(model, optimizer, records, rewards, temperature=1.0, batch_size=32):
    if not records:
        return None  # Explicitly no optimizer.step; a real zero loss still counts.
    model.eval()
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    # Chunked backward is exactly the full-episode mean, without a huge graph.
    for start in range(0, len(records), batch_size):
        part = records[start:start+batch_size]
        states, masks, actions, colors = zip(*part)
        import numpy as np
        x = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        dist = distribution(model(x), np.stack(masks), temperature)
        returns = torch.tensor([rewards.get((c, a), rewards[c]) for c, a in zip(colors, actions)], device=device)
        loss = -(dist.log_prob(torch.tensor(actions, device=device)) * returns).sum() / len(records)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite REINFORCE loss')
        loss.backward()
        total += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return total


def _atomic_copy(source, destination):
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix='.latest-', suffix='.tmp')
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.manifest-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _archive_boards(output, epoch_dir, episodes):
    target = epoch_dir / 'boards.zip'
    fd, temporary = tempfile.mkstemp(dir=epoch_dir, prefix='.boards-', suffix='.tmp')
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            for episode in episodes:
                directory = output / 'games' / f'episode-{episode:08d}'
                for name in ('board.png', 'result.json'):
                    archive.write(directory / name, f'{directory.name}/{name}')
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _upload_completed_epoch(output, project, run_id, epoch_dir):
    """Keep each epoch's data paired with its own immutable model on retries."""
    ready = json.loads((epoch_dir / 'ready.json').read_text())
    metadata = json.loads((epoch_dir / 'metadata.json').read_text())
    if (epoch_dir.name != f"epoch-{int(ready['epoch']):06d}"
            or epoch_dir.parent.resolve() != (output / 'epochs').resolve()
            or metadata.get('epoch') != ready['epoch']
            or metadata.get('run_id') != run_id or ready.get('run_id') != run_id
            or not metadata.get('complete')
            or set(ready['sha256']) != {'policy.pth', 'value.npz', 'metadata.json', 'boards.zip'}):
        raise ValueError('Epoch model/data/run manifest pairing does not match')
    for filename, digest in ready['sha256'].items():
        if _digest(epoch_dir / filename) != digest:
            raise ValueError(f'Completed epoch file changed: {epoch_dir / filename}')
    status_path = epoch_dir / 'uploads.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if not status.get('value'):
        upload_epoch_value(epoch_dir, run_id)
        status['value'] = True
        _write_json(status_path, status)
    if not status.get('model'):
        upload_training(output, project, run_id, epoch=ready['epoch'])
        status['model'] = True
        _write_json(status_path, status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='local policy; default: Hugging Face RL policy, then supervised fallback')
    parser.add_argument('--output', required=True, help='new run directory (existing directory for --upload-only)')
    parser.add_argument('--cache-dir', type=Path,
                        default=Path(tempfile.gettempdir()) / 'go2-rl-hub-cache',
                        help='download cache on local disk; keep separate from a Drive output directory')
    parser.add_argument('--epochs', type=int, default=1, help='additional completed epochs to train')
    parser.add_argument('--updates-per-epoch', type=int, default=4096, help='exact optimizer steps per epoch')
    parser.add_argument('--max-games-per-epoch', type=int, help='optional attempt cap; incomplete epochs are never published')
    parser.add_argument('--max-moves', type=int, default=800)
    parser.add_argument('--komi', type=float, default=7.5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--opponent-mode', choices=('history', 'self', 'pool'), default='history',
                        help='default: random older HF policy; self: both sides; pool: in-run snapshots')
    parser.add_argument('--bootstrap-opponent', action='store_true',
                        help='if HF history is empty, explicitly archive the supervised baseline as first opponent')
    parser.add_argument('--pool-size', type=int, default=0, help='only used with --opponent-mode pool')
    parser.add_argument('--resume', action='store_true', help='restore RL optimizer and episode counter')
    parser.add_argument('--no-upload', action='store_true', help='keep game records/model/code local only')
    parser.add_argument('--upload-only', action='store_true', help='retry uploads from --output without training')
    args = parser.parse_args(argv)
    args.cache_dir = str(args.cache_dir)
    output = Path(args.output)
    project = Path(__file__).resolve().parent.parent
    if args.upload_only:
        if args.no_upload:
            parser.error('--upload-only conflicts with --no-upload')
        run = json.loads((output / 'run.json').read_text())
        epochs = [path for path in sorted((output / 'epochs').glob('epoch-*'))
                  if (path / 'ready.json').exists()]
        if not epochs:
            parser.error('no completed epochs to upload; partial epochs remain local')
        for epoch_dir in epochs:
            _upload_completed_epoch(output, project, run['run_id'], epoch_dir)
            _atomic_copy(epoch_dir / 'policy.pth', output / 'last.pth')
        print(f'Uploads complete: {output}', flush=True)
        return
    import math
    if (args.epochs < 1 or args.updates_per_epoch < 1 or args.max_moves < 2 or args.pool_size < 0
            or (args.max_games_per_epoch is not None and args.max_games_per_epoch < 1)
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.temperature) or args.temperature <= 0
            or not math.isfinite(args.komi)):
        parser.error('invalid training configuration')
    if args.bootstrap_opponent and (args.no_upload or args.opponent_mode != 'history'):
        parser.error('--bootstrap-opponent requires history mode with uploads enabled')
    if args.opponent_mode == 'pool' and args.pool_size < 1:
        parser.error('pool mode requires --pool-size >= 1')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output.mkdir(parents=True, exist_ok=False)
    history = None
    candidates = []
    if args.opponent_mode == 'history':
        latest, candidates = discover_history()
        if args.checkpoint is not None or latest is None:
            path, source = resolve_checkpoint(args.checkpoint, cache_dir=args.cache_dir)
            with open(path, 'rb') as stream:
                source['identity'] = 'sha256:' + hashlib.file_digest(stream, 'sha256').hexdigest()
            candidates = [item for item in candidates if item['identity'] != source['identity']]
        else:
            source = latest
            path = download_version(source, cache_dir=args.cache_dir)
        if not candidates and args.bootstrap_opponent:
            candidates = [bootstrap_opponent(source, cache_dir=args.cache_dir)]
        history = HistoricalOpponentPool(candidates, cache_dir=args.cache_dir)
        print(f'Historical opponent pool: {len(history.candidates)} distinct versions; '
              'one uniform random draw per game; pool fixed for this run', flush=True)
    else:
        path, source = resolve_checkpoint(args.checkpoint, cache_dir=args.cache_dir)
    print(f'Loading policy: {source}', flush=True)
    model, checkpoint = load_policy(path, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    initial = 0
    initial_epoch = 0
    total_updates = 0
    if args.resume:
        if 'episode' not in checkpoint or 'optimizer_state_dict' not in checkpoint:
            parser.error('--resume requires an RL checkpoint')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial = checkpoint['episode']
        initial_epoch = int(checkpoint.get('epoch', 0))
        total_updates = int(checkpoint.get('total_updates', 0))
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
    shutil.copy2(path, output / 'initial.pth')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:12]
    snapshot_code(output, project)
    run = dict(run_id=run_id, source=source, source_code='code', config=vars(args),
               historical_opponents=candidates)
    if history is not None:
        run['opponent_sampling'] = dict(method='uniform_with_replacement',
                                       distinct_versions=len(history.candidates),
                                       refresh='at_run_start')
    (output / 'run.json').write_text(json.dumps(run, ensure_ascii=False, indent=2) + '\n')
    pool = [copy.deepcopy(model).eval().requires_grad_(False)] if args.opponent_mode == 'pool' else []
    opponent = None
    episode = initial
    for epoch in range(initial_epoch + 1, initial_epoch + args.epochs + 1):
        data = EpochValueDataset(output, run_id)
        updates = 0
        ready = False
        try:
            while updates < args.updates_per_epoch:
                if args.max_games_per_epoch is not None and data.attempted_games >= args.max_games_per_epoch:
                    raise RuntimeError(f'Epoch {epoch} incomplete: {updates}/{args.updates_per_epoch} '
                                       'updates; --max-games-per-epoch reached')
                episode += 1
                env = GoAdapter(args.komi, args.max_moves)
                sampler = ReservoirPositionSampler(random.Random(args.seed + episode * 1_000_003))
                opponent = None  # Release last GPU opponent before loading another.
                opponent_source = None
                if history is not None:
                    opponent, opponent_source = history.sample(device=args.device)
                elif pool:
                    opponent = random.choice(pool)
                    opponent_source = dict(kind='in_run_pool')
                learner_color = 1 if episode % 2 else -1
                print(f'epoch={epoch} game={episode} updates={updates}/{args.updates_per_epoch} '
                      f'learner_color={learner_color} opponent={opponent_source}', flush=True)
                records = rollout(model, env, opponent, learner_color, args.temperature, sampler=sampler)
                if sampler.sample is None:
                    raise RuntimeError('rollout did not provide a sampled pre-action position')
                try:
                    metadata = env.result()
                except Exception as exc:
                    metadata = dict(episode=episode, moves=env.moves, winner=None, value_label=None,
                                    result='SCORING ERROR', result_source=None, skip_reason='scoring_error',
                                    error_type=type(exc).__name__, history=env.history)
                    position = sampler.sample
                    save_game(output, episode, position['board'], position['state'], metadata)
                    data.add_game(episode, position, metadata)
                    raise
                metadata.pop('history', None)  # Export one position/result, not the full move trace.
                loss = None if env.skip_reason else reinforce_update(
                    model, optimizer, records, env.rewards(), args.temperature)
                did_update = loss is not None  # 0.0 is a successful step too.
                if did_update:
                    updates += 1
                    total_updates += 1
                position = sampler.sample
                metadata.update(epoch=epoch, episode=episode, run_id=run_id, learner_source=source,
                                opponent_source=opponent_source, learner_color=learner_color,
                                did_update=did_update, loss=loss, updates_in_epoch=updates,
                                total_updates=total_updates, sample_move_index=position['move_index'],
                                sample_player=position['player'], board_kind='random_pre_action',
                                terminal_encoding_player=metadata.get('encoding_player'),
                                encoding_player=position['player'])
                save_game(output, episode, position['board'], position['state'], metadata)
                data.add_game(episode, position, metadata)
                print(f'epoch={epoch} game={episode} updates={updates}/{args.updates_per_epoch} '
                      f'moves={env.moves} result={metadata["result"]} '
                      f'skipped={env.skip_reason} loss={loss}', flush=True)
            epoch_dir = data.save(epoch, updates, complete=True)
            _archive_boards(output, epoch_dir, [game['episode'] for game in data.games])
            # Serialize trained weights once per completed epoch; last.pth is an atomic copy.
            save_checkpoint(epoch_dir / 'policy.pth', model, optimizer, episode, vars(args),
                            epoch=epoch, total_updates=total_updates)
            _write_json(epoch_dir / 'ready.json', dict(epoch=epoch, run_id=run_id, total_updates=total_updates,
                        sha256={name: _digest(epoch_dir / name) for name in
                                ('policy.pth', 'value.npz', 'metadata.json', 'boards.zip')}))
            ready = True
            _atomic_copy(epoch_dir / 'policy.pth', output / 'last.pth')
            if pool:
                pool.append(copy.deepcopy(model).eval().requires_grad_(False))
                pool = pool[-args.pool_size:]
            print(f'epoch={epoch} complete updates={updates} games={data.attempted_games} '
                  f'value_samples={data.sample_count}', flush=True)
            if not args.no_upload:
                _upload_completed_epoch(output, project, run_id, epoch_dir)
        except BaseException:
            if not ready:
                # Preserve sampled data for inspection, not as a completed training epoch.
                data.save(epoch, updates, complete=False)
            raise
    print(f'Run saved: {output}', flush=True)


if __name__ == '__main__':
    main()
