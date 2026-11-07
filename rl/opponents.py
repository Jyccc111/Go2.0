"""Pinned Hugging Face policy versions for random, frozen historical opponents.

Discovery reads metadata only. Weights are downloaded lazily for the selected
opponent; no downloads, training or remote writes occur on import.
"""
import random
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RevisionNotFoundError

from .hub import MODEL_REPO, _policy_file
from .policy import load_policy


class NoHistoricalOpponentError(ValueError):
    """The repository contains no distinct checkpoint older than the learner."""


NO_HISTORY_MESSAGE = (
    'Go_RL_policy has no distinct historical policy checkpoint. The root last.pth '
    'and its run-scoped copy are the same model and cannot be historical opponents. '
    'Add an older checkpoint, or use --bootstrap-opponent explicitly to archive '
    'the initial supervised policy as a predecessor before training.'
)


def _field(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _identity(sibling):
    sha = _field(getattr(sibling, 'lfs', None), 'sha256')
    if sha:
        return 'sha256:' + sha
    blob = getattr(sibling, 'blob_id', None)
    if blob:
        return 'blob:' + blob
    raise ValueError(f'Missing content identity for checkpoint {sibling.rfilename}; '
                     'model_info(files_metadata=True) is required')


def _checkpoints(info):
    return [sibling for sibling in (info.siblings or [])
            if Path(sibling.rfilename).suffix.lower() in ('.pth', '.pt', '.ckpt')]


def _source(repo, revision, sibling):
    return dict(kind='huggingface', repo_id=repo, revision=revision,
                filename=sibling.rfilename, identity=_identity(sibling))


def _head_info(api, repo):
    try:
        return api.model_info(repo_id=repo, files_metadata=True)
    except RevisionNotFoundError:
        # Only a genuinely accessible empty repository may have no main revision.
        info = api.repo_info(repo_id=repo, repo_type='model')
        if info.sha is None:
            return None
        raise


def discover_history(*, api=None, model_repo=MODEL_REPO):
    """Return (latest learner source or None, unique historical sources).

    HEAD and its ancestor history are pinned at discovery. Current retained
    snapshots and previous revisions are deduplicated by checkpoint file content;
    the learner's content is excluded everywhere, including its archived copy.
    An empty list is returned for no history so the caller can give an explicit
    bootstrap choice; authentication/network errors always propagate.
    """
    api = api or HfApi()
    head = _head_info(api, model_repo)
    if head is None:
        return None, []
    weights = _checkpoints(head)
    if not head.sha:
        raise ValueError('Cannot pin model repository without a commit SHA')
    filename = _policy_file([s.rfilename for s in weights])
    sibling = next((s for s in weights if s.rfilename == filename), None)
    learner = _source(model_repo, head.sha, sibling) if sibling is not None else None
    seen = {learner['identity']} if learner is not None else set()
    opponents = []

    def collect(info):
        for checkpoint in sorted(_checkpoints(info), key=lambda s: s.rfilename):
            version = _source(model_repo, info.sha, checkpoint)
            if version['identity'] not in seen:
                seen.add(version['identity'])
                opponents.append(version)

    collect(head)
    commits = api.list_repo_commits(repo_id=model_repo, repo_type='model', revision=head.sha)
    for commit in commits:
        revision = commit.commit_id
        if revision == head.sha:
            continue
        info = api.model_info(repo_id=model_repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError('Historical model metadata revision did not match requested commit')
        collect(info)
    return learner, opponents


def download_version(source, cache_dir=None):
    """Download the exact pinned file represented by a discovery result."""
    if not source.get('revision'):
        raise ValueError('A pinned commit revision is required')
    return hf_hub_download(repo_id=source['repo_id'], filename=source['filename'],
                           revision=source['revision'], repo_type='model',
                           cache_dir=cache_dir)


class HistoricalOpponentPool:
    """Uniform choice across distinct pinned policies; only selected weights load."""

    def __init__(self, candidates, cache_dir=None):
        if not candidates:
            raise NoHistoricalOpponentError(NO_HISTORY_MESSAGE)
        # Defensive deduplication when callers build a candidate list themselves.
        unique = {c['identity']: dict(c) for c in candidates}
        self.candidates = list(unique.values())
        self.cache_dir = cache_dir

    def sample(self, device='cpu', rng=None):
        """Return one frozen policy and JSON-safe provenance for this episode."""
        rng = rng or random
        source = dict(rng.choice(self.candidates))
        path = download_version(source, cache_dir=self.cache_dir)
        model, _ = load_policy(path, device)
        model.eval()
        model.requires_grad_(False)
        return model, source
