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
from .game import GoAdapter, PASS, MIN_TRAIN_MOVES
from .policy import distribution, load_policy, save_checkpoint
from .hub import resolve_checkpoint, save_game, upload_training, snapshot_code, bootstrap_opponent

from .opponents import discover_history, download_version, HistoricalOpponentPool
from .value_data import ReservoirPositionSampler, EpochValueDataset, upload_epoch_value


def rollout(model, env, opponent=None, learner_color=1, temperature=1.0, sampler=None):
    model.eval()  # Freeze BatchNorm statistics; eval still allows gradients.
    if opponent is not None:
        opponent.eval()
    records = []
    pass_stats = {color: dict(positions=0, probability_sum=0.0, late_positions=0,
                             late_probability_sum=0.0, late_max_probability=0.0)
                  for color in (1, -1)}
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
            dist = distribution(logits, mask, temperature)
            probability = float(dist.probs[PASS].item())
            action = int(dist.sample().item())
        stats = pass_stats[color]
        stats['positions'] += 1
        stats['probability_sum'] += probability
        if env.moves >= MIN_TRAIN_MOVES:
            stats['late_positions'] += 1
            stats['late_probability_sum'] += probability
            stats['late_max_probability'] = max(stats['late_max_probability'], probability)
        if learner:
            records.append((state, mask, action, color))
        env.step(action)
    env.pass_policy = {}
    for color, stats in pass_stats.items():
        env.pass_policy[str(color)] = dict(
            positions=stats['positions'], late_positions=stats['late_positions'],
            mean_probability=stats['probability_sum'] / stats['positions'] if stats['positions'] else None,
            late_mean_probability=stats['late_probability_sum'] / stats['late_positions']
            if stats['late_positions'] else None,
            late_max_probability=stats['late_max_probability'] if stats['late_positions'] else None)
    return records


def action_returns(records, rewards, outcome=None, terminal_pass_bonus=0.0):
    """Experimental, once-only bonus to the winner's final terminal pass.

    No bonus for losing, early, nonterminal, resigned or truncated games. Earlier
    passes by the winner keep the ordinary terminal reward. This is not a
    policy-invariant shaping rule and is disabled by default.
    """
    import math
    if not math.isfinite(terminal_pass_bonus) or not 0 <= terminal_pass_bonus <= 0.1:
        raise ValueError('terminal_pass_bonus must be between 0 and 0.1')
    values = [rewards.get((color, action), rewards[color]) for _, _, action, color in records]
    if terminal_pass_bonus == 0:
        return values
    if outcome is None:
        raise ValueError('terminal pass bonus requires an adjudicated outcome')
    if (outcome.get('termination') != 'two_passes' or outcome.get('moves', 0) < MIN_TRAIN_MOVES
            or outcome.get('skip_reason') or outcome.get('truncated')
            or outcome.get('result_source') != 'Game.cal_result'):
        return values
    for index in range(len(records) - 1, -1, -1):
        if records[index][3] == outcome.get('winner'):
            if records[index][2] == PASS:
                values[index] += terminal_pass_bonus
            break
    return values


def reinforce_update(model, optimizer, records, rewards, temperature=1.0, batch_size=32,
                     *, outcome=None, terminal_pass_bonus=0.0):
    if not records:
        return None  # Explicitly no optimizer.step; a real zero loss still counts.
    model.eval()
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    values = action_returns(records, rewards, outcome, terminal_pass_bonus)
    total = 0.0
    # Chunked backward is exactly the full-episode mean, without a huge graph.
    for start in range(0, len(records), batch_size):
        part = records[start:start+batch_size]
        states, masks, actions, colors = zip(*part)
        import numpy as np
        x = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        dist = distribution(model(x), np.stack(masks), temperature)
        returns = torch.tensor(values[start:start+len(part)], device=device)
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
    parser.add_argument('--max-moves', type=int, default=500,
                        help='both colors combined; unfinished games at this limit are skipped')
    parser.add_argument('--terminal-pass-bonus', type=float, default=0.0,
                        help='experimental 0..0.1 bonus to winner final pass after scored two-pass end; default off')
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
            or not math.isfinite(args.terminal_pass_bonus) or not 0 <= args.terminal_pass_bonus <= 0.1
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
                    model, optimizer, records, env.rewards(), args.temperature,
                    outcome=metadata, terminal_pass_bonus=args.terminal_pass_bonus)
                did_update = loss is not None  # 0.0 is a successful step too.
                pass_bonus_awarded = 0.0
                if did_update and args.terminal_pass_bonus:
                    base = action_returns(records, env.rewards())
                    adjusted = action_returns(records, env.rewards(), metadata, args.terminal_pass_bonus)
                    pass_bonus_awarded = sum(a - b for a, b in zip(adjusted, base))
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
                                encoding_player=position['player'],
                                pass_policy=getattr(env, 'pass_policy', None),
                                terminal_pass_bonus=args.terminal_pass_bonus,
                                terminal_pass_bonus_awarded=pass_bonus_awarded)
                save_game(output, episode, position['board'], position['state'], metadata)
                data.add_game(episode, position, metadata)
                print(f'epoch={epoch} game={episode} updates={updates}/{args.updates_per_epoch} '
                      f'moves={env.moves} result={metadata["result"]} '
                      f'skipped={env.skip_reason} loss={loss}', flush=True)
                stats = (metadata['pass_policy'] or {}).get(str(learner_color), {})
                late_probability = stats.get('late_mean_probability')
                late_text = f'{late_probability:.4f}' if late_probability is not None else 'n/a'
                print(f'passes={metadata.get("pass_counts")} '
                      f'learner_late_pass_probability={late_text} '
                      f'terminal_pass_bonus_awarded={pass_bonus_awarded:.3f}', flush=True)
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
            limit_games = sum(game.get('skip_reason') == 'move_limit' for game in data.games)
            two_pass_games = sum(game.get('termination') == 'two_passes' for game in data.games)
            print(f'epoch={epoch} endgames two_passes={two_pass_games} '
                  f'move_limit={limit_games}/{data.attempted_games} '
                  f'({limit_games / data.attempted_games:.1%}) '
                  f'mean_moves={sum(game["moves"] for game in data.games) / data.attempted_games:.1f}', flush=True)
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
