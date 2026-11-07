"""Checkpoint compatibility and the exact masked sampling distribution."""
import os
import io
import hashlib
import pickle
from pathlib import Path
import tempfile
import torch
from torch.distributions import Categorical
from Train_net import Policy_network


class CheckpointFileError(ValueError):
    """A checkpoint is unreadable or has unexpected downloaded content."""


def verify_checkpoint_file(path, identity):
    """Verify pinned LFS content before loading; do not substitute another model."""
    if not identity or not identity.startswith('sha256:'):
        return
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != identity.split(':', 1)[1]:
        raise CheckpointFileError(
            f'Checkpoint checksum mismatch: {path}. Re-download into a local disk cache '
            '(in Colab use --cache-dir /content/go2-rl-hub-cache, not Google Drive).')


def load_policy(path, device='cpu'):
    # Read once, then use a seekable in-memory buffer rather than Drive/FUSE seeks.
    data = Path(path).read_bytes()
    if not data:
        raise CheckpointFileError(f'Empty checkpoint: {path}')
    if len(data) < 4096 and data.startswith((b'../', b'./')):
        raise CheckpointFileError(
            f'Checkpoint contains a symlink target instead of weights: {path}. '
            'Use a local disk --cache-dir, not a Google Drive cache.')
    if data.startswith(b'version https://git-lfs.github.com/spec/'):
        raise CheckpointFileError(f'Checkpoint is a Git LFS pointer, not downloaded weights: {path}')
    try:
        checkpoint = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    except (IndexError, EOFError, RuntimeError, pickle.UnpicklingError) as exc:
        raise CheckpointFileError(
            f'Cannot read checkpoint {path}: invalid, incomplete or incompatible weights. '
            'Re-download the same version to a local disk cache; the model has not been reset.') from exc
    state = checkpoint.get('model_state_dict', checkpoint)
    state = {k.removeprefix('network.'): v for k, v in state.items()}
    blocks = {k.split('.')[0] for k in state if k.endswith('.conv1.weight')}
    model = Policy_network(size=int(state['0.weight'].shape[0]), num_blocks=len(blocks))
    model.network.load_state_dict(state, strict=True)
    return model.to(device), checkpoint


def distribution(logits, mask, temperature=1.0):
    if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError('temperature must be finite and positive')
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if logits.shape != mask.shape or not mask.any(dim=-1).all():
        raise ValueError('mask shape mismatch or no legal actions')
    if not torch.isfinite(logits).all():
        raise FloatingPointError('nonfinite policy logits')
    return Categorical(logits=(logits / temperature).masked_fill(~mask, -torch.inf))


def save_checkpoint(path, model, optimizer, episode, config, *, epoch=None, total_updates=None):
    """Atomic local replace; never upload or overwrite the source policy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(model_state_dict={k: v.detach().cpu() for k,v in model.network.state_dict().items()},
                   optimizer_state_dict=optimizer.state_dict(), episode=episode,
                   size=model.size, num_blocks=model.num_blocks, config=config,
                   torch_rng_state=torch.get_rng_state())
    if epoch is not None:
        payload['epoch'] = int(epoch)
        payload['total_updates'] = int(total_updates)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.policy-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
