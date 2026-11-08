"""Content-addressed policy versions in one directory, independent of run dates."""
import hashlib
import re
from pathlib import Path

_FLAT = re.compile(r'^models/(?:epoch-([0-9]+)|policy)-([0-9a-f]{64}|[0-9a-f]{40})\.pth$')
_OLD_EPOCH = re.compile(r'^runs/([A-Za-z0-9._-]+)/epochs/epoch-([0-9]+)/policy\.pth$')


def version_details(filename):
    """Return (epoch, digest) for flat files or (epoch, None) for old epochs."""
    match = _FLAT.fullmatch(filename)
    if match:
        number, digest = match.groups()
        return (int(number) if number else None), digest
    match = _OLD_EPOCH.fullmatch(filename)
    if match:
        return int(match.group(2)), None
    return None


def version_path(digest, epoch=None):
    if not re.fullmatch(r'[0-9a-f]{64}|[0-9a-f]{40}', digest):
        raise ValueError('Expected a SHA-256 or Git blob content digest')
    if epoch is not None and (int(epoch) != epoch or epoch < 1):
        raise ValueError('epoch must be positive')
    stem = f'epoch-{int(epoch):06d}' if epoch is not None else 'policy'
    return f'models/{stem}-{digest}.pth'


def local_version_path(path, epoch=None):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return version_path(digest, epoch)
