"""Checkpoint compatibility and the exact masked sampling distribution."""
import os
from pathlib import Path
import tempfile
import torch
from torch.distributions import Categorical
from Train_net import Policy_network


def load_policy(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
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


def save_checkpoint(path, model, optimizer, episode, config):
    """Atomic local replace; never upload or overwrite the source policy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(model_state_dict={k: v.detach().cpu() for k,v in model.network.state_dict().items()},
                   optimizer_state_dict=optimizer.state_dict(), episode=episode,
                   size=model.size, num_blocks=model.num_blocks, config=config,
                   torch_rng_state=torch.get_rng_state())
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
