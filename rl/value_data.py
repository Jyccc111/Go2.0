"""One uniformly sampled intermediate position per game, batched by policy epoch.

This module never trains a value network and does no network I/O on import.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile

import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi

DATA_REPO = 'Jycccc111/Go_value_data'
SAMPLING = ('uniform reservoir over all live pre-action positions; '
            'both colors; initial position included and terminal position excluded')


def _position(state, board, move_index, player):
    state, board = np.asarray(state), np.asarray(board)
    if state.shape != (8, 19, 19) or board.shape != (19, 19):
        raise ValueError('expected state (8,19,19) and board (19,19)')
    if not np.isfinite(state).all() or not np.isfinite(board).all():
        raise ValueError('sampled position contains nonfinite values')
    if int(move_index) != move_index or move_index < 0 or player not in (-1, 1):
        raise ValueError('sample requires move_index >= 0 and player in {-1,1}')
    return dict(state=state.astype(np.int16, copy=True),
                board=board.astype(np.int8, copy=True),
                move_index=int(move_index), player=int(player))


class ReservoirPositionSampler:
    """Keep one position with uniform probability, using constant board memory.

    The caller offers all live pre-action positions, including move_index=0.
    Use a dedicated random.Random instance to avoid changing policy/opponent RNG.
    """
    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random.Random()
        self.seen = 0
        self.sample = None

    def consider(self, state, board, move_index, player):
        if move_index < 0:
            raise ValueError('move_index must be nonnegative')
        self.seen += 1
        if self.rng.randrange(self.seen) == 0:
            self.sample = _position(state, board, move_index, player)
        return self.sample


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported metadata type: {type(value).__name__}')


def _atomic_write(path, writer):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name + '-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class EpochValueDataset:
    """Accumulate at most one labelled position from each completed game.

    Attempted games and effective policy updates are independent counts. Games
    with no position, no known result or a skip reason never fabricate a label.
    """
    def __init__(self, output, run_id=None):
        self.output = Path(output)
        self.run_id = run_id
        self.games = []
        self.samples = []
        self._episodes = set()

    @property
    def sample_count(self):
        return len(self.samples)

    @property
    def attempted_games(self):
        return len(self.games)

    def add_game(self, episode, sample, outcome_metadata):
        episode = int(episode)
        if episode < 1 or episode in self._episodes:
            raise ValueError('episode must be positive and occur only once per epoch dataset')
        outcome = copy.deepcopy({key: value for key, value in outcome_metadata.items()
                                 if key != 'history'})
        label = outcome.get('value_label')
        if label is not None and label not in (0, 1):
            raise ValueError('value_label must be 1 (Black), 0 (White), or None')
        skipped = outcome.get('skip_reason') or ('truncated' if outcome.get('truncated') else None)
        if skipped:
            exclusion = 'skipped'
        elif label is None:
            exclusion = 'unlabelled'
        elif sample is None:
            exclusion = 'no_eligible_position'
        else:
            exclusion = None
        position = None
        if exclusion is None:
            position = _position(sample['state'], sample['board'],
                                 sample['move_index'], sample['player'])
        game = dict(outcome, episode=episode, value_sample_included=position is not None,
                    value_sample_exclusion=exclusion)
        if position is not None:
            game['sample_move_index'] = position['move_index']
            game['sample_player'] = position['player']
            position.update(episode=episode, result=int(label))
            self.samples.append(position)
        self._episodes.add(episode)
        self.games.append(game)
        return position is not None

    def save(self, epoch, updates, complete=True):
        if int(epoch) != epoch or epoch < 1 or int(updates) != updates or updates < 0:
            raise ValueError('positive epoch and nonnegative integer updates required')
        path = self.output / 'epochs' / f'epoch-{int(epoch):06d}'
        path.mkdir(parents=True, exist_ok=True)
        exclusions = [game['value_sample_exclusion'] for game in self.games]
        manifest = dict(schema_version=1, run_id=self.run_id, epoch=int(epoch),
                        updates=int(updates), complete=bool(complete), sampling=SAMPLING,
                        attempted_games=self.attempted_games, sample_count=self.sample_count,
                        skipped_games=exclusions.count('skipped'),
                        unlabelled_games=exclusions.count('unlabelled'),
                        no_eligible_position_games=exclusions.count('no_eligible_position'),
                        games=self.games)
        encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                             allow_nan=False, default=_json_default)
        samples = self.samples
        arrays = dict(
            states=np.stack([s['state'] for s in samples]) if samples else np.empty((0,8,19,19), dtype=np.int16),
            results=np.asarray([s['result'] for s in samples], dtype=np.int8),
            boards=np.stack([s['board'] for s in samples]) if samples else np.empty((0,19,19), dtype=np.int8),
            episodes=np.asarray([s['episode'] for s in samples], dtype=np.int64),
            move_indices=np.asarray([s['move_index'] for s in samples], dtype=np.int32),
            players=np.asarray([s['player'] for s in samples], dtype=np.int8),
            metadata=np.asarray(encoded))
        _atomic_write(path / 'value.npz', lambda stream: np.savez_compressed(stream, **arrays))
        manifest['dataset_sha256'] = _sha256(path / 'value.npz')
        payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2,
                             allow_nan=False, default=_json_default).encode('utf-8') + b'\n'
        # The manifest is the last commit marker; its hash detects interrupted pairs.
        _atomic_write(path / 'metadata.json', lambda stream: stream.write(payload))
        return path


def upload_epoch_value(epoch_dir, run_id, *, api=None, data_repo=DATA_REPO):
    """One dataset commit per completed, nonempty epoch; no per-game uploads."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', run_id):
        raise ValueError('run_id must be a simple unique filename component')
    path = Path(epoch_dir)
    manifest = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise ValueError('incomplete epoch datasets cannot be uploaded')
    if manifest['dataset_sha256'] != _sha256(path / 'value.npz'):
        raise ValueError('epoch dataset checksum does not match its manifest')
    if manifest.get('sample_count', 0) == 0:
        return None
    expected = f"epoch-{int(manifest['epoch']):06d}"
    if path.name != expected:
        raise ValueError('epoch directory and manifest do not match')
    api = api if api is not None else HfApi()
    operations = [CommitOperationAdd(
        path_in_repo=f'runs/{run_id}/epochs/{path.name}/{name}',
        path_or_fileobj=str(path / name)) for name in ('value.npz', 'metadata.json')]
    if (path / 'boards.zip').is_file():
        operations.append(CommitOperationAdd(
            path_in_repo=f'runs/{run_id}/epochs/{path.name}/boards.zip',
            path_or_fileobj=str(path / 'boards.zip')))
    return api.create_commit(repo_id=data_repo, repo_type='dataset', operations=operations,
                             commit_message=f'Add epoch value samples {run_id}/{path.name}')
