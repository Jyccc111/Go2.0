"""One on-policy episode per optimizer update; no retained rollout graphs."""
import argparse
import copy
import json
from datetime import datetime, timezone
import uuid
from pathlib import Path
import random
import torch
from .game import GoAdapter
from .policy import distribution, load_policy, save_checkpoint
from .hub import resolve_checkpoint, save_game, upload_game, upload_training, snapshot_code


def rollout(model, env, opponent=None, learner_color=1, temperature=1.0):
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
        with torch.no_grad():
            logits = actor(torch.as_tensor(state, dtype=torch.float32, device=device)[None])[0]
            action = int(distribution(logits, mask, temperature).sample().item())
        if learner:
            records.append((state, mask, action, color))
        env.step(action)
    return records


def reinforce_update(model, optimizer, records, rewards, temperature=1.0, batch_size=32):
    if not records:
        return 0.0
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='local policy; default: Hugging Face RL policy, then supervised fallback')
    parser.add_argument('--output', required=True, help='new run directory (existing directory for --upload-only)')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--max-moves', type=int, default=800)
    parser.add_argument('--komi', type=float, default=7.5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pool-size', type=int, default=0)
    parser.add_argument('--snapshot-every', type=int, default=20)
    parser.add_argument('--resume', action='store_true', help='restore RL optimizer and episode counter')
    parser.add_argument('--no-upload', action='store_true', help='keep game records/model/code local only')
    parser.add_argument('--upload-only', action='store_true', help='retry uploads from --output without training')
    args = parser.parse_args(argv)
    output = Path(args.output)
    project = Path(__file__).resolve().parent.parent
    if args.upload_only:
        if args.no_upload:
            parser.error('--upload-only conflicts with --no-upload')
        run = json.loads((output / 'run.json').read_text())
        for game_dir in sorted((output / 'games').glob('episode-*')):
            upload_game(game_dir, run['run_id'])
        upload_training(output, project, run['run_id'])
        print(f'Uploads complete: {output}', flush=True)
        return
    import math
    if (args.episodes < 1 or args.max_moves < 2 or args.pool_size < 0 or args.snapshot_every < 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.temperature) or args.temperature <= 0
            or not math.isfinite(args.komi)):
        parser.error('invalid training configuration')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output.mkdir(parents=True, exist_ok=False)
    path, source = resolve_checkpoint(args.checkpoint, cache_dir=output / 'hub-cache')
    print(f'Loading policy: {source}', flush=True)
    model, checkpoint = load_policy(path, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    initial = 0
    if args.resume:
        if 'episode' not in checkpoint or 'optimizer_state_dict' not in checkpoint:
            parser.error('--resume requires an RL checkpoint')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial = checkpoint['episode']
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:12]
    snapshot_code(output, project)
    run = dict(run_id=run_id, source=source, source_code='code', config=vars(args))
    (output / 'run.json').write_text(json.dumps(run, ensure_ascii=False, indent=2) + '\n')
    pool = [copy.deepcopy(model).eval()] if args.pool_size else []
    for episode in range(initial+1, initial+args.episodes+1):
        env = GoAdapter(args.komi, args.max_moves)
        opponent = random.choice(pool) if pool else None
        records = rollout(model, env, opponent, 1 if episode % 2 else -1, args.temperature)
        # Archive every terminal board before updating or attempting remote writes.
        # A scoring error still leaves the raw board and an explicit unknown label.
        try:
            metadata = env.result()
        except Exception as exc:
            save_game(output, episode, env.game.board.board, env.observe(), dict(
                episode=episode, moves=env.moves, winner=None, value_label=None,
                result='SCORING ERROR', result_source=None, skip_reason='scoring_error',
                error_type=type(exc).__name__, history=env.history))
            save_checkpoint(output / 'last.pth', model, optimizer, episode-1, vars(args))
            raise
        metadata.update(episode=episode, run_id=run_id)
        game_dir = save_game(output, episode, env.game.board.board, env.observe(), metadata)
        loss = None if env.skip_reason else reinforce_update(model, optimizer, records, env.rewards(), args.temperature)
        print(f'episode={episode} moves={env.moves} result={metadata["result"]} '
              f'skipped={env.skip_reason} loss={loss}', flush=True)
        save_checkpoint(output / 'last.pth', model, optimizer, episode, vars(args))
        if pool and episode % args.snapshot_every == 0:
            pool.append(copy.deepcopy(model).eval())
            pool = pool[-args.pool_size:]
            save_checkpoint(output / f'policy-{episode}.pth', model, optimizer, episode, vars(args))
        if not args.no_upload:
            upload_game(game_dir, run_id)
    if not args.no_upload:
        upload_training(output, project, run_id)
    print(f'Run saved: {output}', flush=True)


if __name__ == '__main__':
    main()
