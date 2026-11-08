"""Web policy catalog; importing it never downloads weights or starts training."""
import hashlib
from threading import RLock
from time import monotonic

from huggingface_hub import HfApi
from rl.model_store import version_details

POLICY_MODELS = {
    'supervised': {
        'label': '监督学习模型',
        'repo_id': 'Jycccc111/Go-Policy',
        'filename': 'best_policy_network_128.pth',
    },
    'rl': {
        'label': '强化学习模型（最新）',
        'repo_id': 'Jycccc111/Go_RL_policy',
        'filename': 'last.pth',
    },
}
_EPOCH_MODELS = {}
_CATALOG = None
_CATALOG_TIME = 0.0
_CATALOG_LOCK = RLock()


def get_policy_model(value):
    """Accept only built-in choices or server-discovered, pinned checkpoints."""
    if not isinstance(value, str):
        raise ValueError('请选择有效的模型版本。')
    with _CATALOG_LOCK:
        source = POLICY_MODELS.get(value) or _EPOCH_MODELS.get(value)
        if source is None:
            raise ValueError('模型版本不存在，请刷新模型列表后重选。')
        return dict(source)


def validate_policy_model(value):
    get_policy_model(value)
    return value


def policy_model_label(value):
    return get_policy_model(value)['label']


def base_policy_choices():
    return [dict(id=key, **source) for key, source in POLICY_MODELS.items()]


def list_policy_models(*, refresh=False, api=None):
    """Read checkpoint metadata only; keep selected versions stable across refreshes."""
    global _CATALOG, _CATALOG_TIME
    with _CATALOG_LOCK:
        if not refresh and _CATALOG is not None and monotonic() - _CATALOG_TIME < 60:
            return [dict(row) for row in _CATALOG]
        api = api or HfApi()
        info = api.model_info(repo_id=POLICY_MODELS['rl']['repo_id'], files_metadata=True)
        if not info.sha:
            raise ValueError('强化学习仓库尚无可读取的版本。')
        versions = {}
        # Flat files take precedence over old copies. Preserve compatibility
        # with repositories not yet organized, without showing either twice.
        siblings = sorted(info.siblings or [], key=lambda s: not s.rfilename.startswith('models/'))
        for sibling in siblings:
            parsed = version_details(sibling.rfilename)
            if parsed is None:
                continue
            epoch, digest = parsed
            if epoch is not None and epoch < 1:
                continue
            lfs = getattr(sibling, 'lfs', None)
            sha = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
            blob = getattr(sibling, 'blob_id', None)
            identity = 'sha256:' + sha if sha else ('blob:' + blob if blob else None)
            # Content identity keeps the option stable when unrelated commits occur.
            content = identity or sibling.rfilename + '@' + info.sha
            key = 'rl-epoch-' + hashlib.sha256(f'{epoch}\0{content}'.encode()).hexdigest()[:24]
            short = (identity.split(':', 1)[1] if identity else digest or info.sha)[:8]
            label = f'强化学习 Epoch {epoch}（{short}）' if epoch is not None else f'历史模型（{short}）'
            run_id = sibling.rfilename.split('/')[1] if sibling.rfilename.startswith('runs/') else None
            source = dict(label=label,
                          repo_id=POLICY_MODELS['rl']['repo_id'], filename=sibling.rfilename,
                          revision=info.sha, identity=identity, epoch=epoch, run_id=run_id, version=True)
            # Old selected options remain valid, with their original immutable commit.
            frozen = _EPOCH_MODELS.setdefault(key, source)
            versions.setdefault((epoch, content), dict(id=key, **frozen))
        epoch_contents = {row['identity'] for row in versions.values() if row['epoch'] is not None}
        rows = [row for row in versions.values()
                if row['epoch'] is not None or row['identity'] not in epoch_contents]
        rows.sort(key=lambda row: (row['epoch'] or 0, row['id']), reverse=True)
        _CATALOG = base_policy_choices() + rows
        _CATALOG_TIME = monotonic()
        return [dict(row) for row in _CATALOG]
