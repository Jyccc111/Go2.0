"""Explicit Hugging Face I/O and dependency-free per-game artifacts.

Nothing downloads or uploads on import. Upload inputs use an explicit allowlist.
"""
import json
from pathlib import Path
import re
import struct
import zlib
import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import RevisionNotFoundError

MODEL_REPO = 'Jycccc111/Go_RL_policy'
FALLBACK_REPO = 'Jycccc111/Go-Policy'
DATA_REPO = 'Jycccc111/Go_value_data'
SOURCE_FILES = (
    'Game.py', 'Train_net.py', 'Reinforcement_learning.py', 'README_RL.md',
    'requirements-rl.txt', 'rl/__init__.py', 'rl/game.py', 'rl/policy.py',
    'rl/train.py', 'rl/hub.py', 'rl/opponents.py', 'tests/test_rl.py',
    'tests/test_rl_hub.py', 'tests/test_rl_opponents.py',
)


def _repo_files(api, repo):
    try:
        return api.list_repo_files(repo_id=repo, repo_type='model')
    except RevisionNotFoundError:
        # An accessible just-created repo can have no main revision. A missing
        # or private repository/network error must not silently select fallback.
        info = api.repo_info(repo_id=repo, repo_type='model')
        if info.sha is None:
            return []
        raise


def _policy_file(files):
    for candidate in ('last.pth', 'best_policy_network_128.pth',
                      'best_policynetwork_128.pth', 'policy.pth'):
        if candidate in files:
            return candidate
    candidates = sorted(f for f in files if Path(f).suffix.lower() in ('.pth','.pt','.ckpt'))
    roots = [f for f in candidates if '/' not in f]
    return roots[0] if roots else (candidates[-1] if candidates else None)


def resolve_checkpoint(local=None, *, api=None, model_repo=MODEL_REPO,
                       fallback_repo=FALLBACK_REPO, fallback_filename=None,
                       cache_dir=None):
    """Fall back only after listing an accessible RL repo without weights."""
    if local is not None:
        path = Path(local).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path), {'kind':'local','path':str(path)}
    api = api or HfApi()
    filename = _policy_file(_repo_files(api,model_repo))
    repo = model_repo
    if filename is None:
        repo = fallback_repo
        files = _repo_files(api,repo)
        options = ([fallback_filename] if fallback_filename else
                   ['best_policy_network_128.pth','best_policynetwork_128.pth',
                    'best_policynetwork_128','best_policy_network_128'])
        filename = next((name for name in options if name in files),None)
        if filename is None:
            raise FileNotFoundError(f'No requested best policy checkpoint in {repo}; tried {options}')
    path = hf_hub_download(repo_id=repo,filename=filename,repo_type='model',cache_dir=cache_dir)
    return path, {'kind':'huggingface','repo_id':repo,'filename':filename}


# Compact 5x7 bitmap font: board screenshots need no extra imaging dependency.
_FONT_ROWS = {
 'A':'01110/10001/10001/11111/10001/10001/10001',
 'B':'11110/10001/10001/11110/10001/10001/11110',
 'C':'01111/10000/10000/10000/10000/10000/01111',
 'D':'11110/10001/10001/10001/10001/10001/11110',
 'E':'11111/10000/10000/11110/10000/10000/11111',
 'F':'11111/10000/10000/11110/10000/10000/10000',
 'G':'01111/10000/10000/10111/10001/10001/01111',
 'H':'10001/10001/10001/11111/10001/10001/10001',
 'I':'11111/00100/00100/00100/00100/00100/11111',
 'J':'00111/00010/00010/00010/10010/10010/01100',
 'K':'10001/10010/10100/11000/10100/10010/10001',
 'L':'10000/10000/10000/10000/10000/10000/11111',
 'M':'10001/11011/10101/10101/10001/10001/10001',
 'N':'10001/11001/10101/10011/10001/10001/10001',
 'O':'01110/10001/10001/10001/10001/10001/01110',
 'P':'11110/10001/10001/11110/10000/10000/10000',
 'Q':'01110/10001/10001/10001/10101/10010/01101',
 'R':'11110/10001/10001/11110/10100/10010/10001',
 'S':'01111/10000/10000/01110/00001/00001/11110',
 'T':'11111/00100/00100/00100/00100/00100/00100',
 'U':'10001/10001/10001/10001/10001/10001/01110',
 'V':'10001/10001/10001/10001/10001/01010/00100',
 'W':'10001/10001/10001/10101/10101/11011/10001',
 'X':'10001/10001/01010/00100/01010/10001/10001',
 'Y':'10001/10001/01010/00100/00100/00100/00100',
 'Z':'11111/00001/00010/00100/01000/10000/11111',
 '0':'01110/10001/10011/10101/11001/10001/01110',
 '1':'00100/01100/00100/00100/00100/00100/01110',
 '2':'01110/10001/00001/00010/00100/01000/11111',
 '3':'11110/00001/00001/01110/00001/00001/11110',
 '4':'00010/00110/01010/10010/11111/00010/00010',
 '5':'11111/10000/10000/11110/00001/00001/11110',
 '6':'01110/10000/10000/11110/10001/10001/01110',
 '7':'11111/00001/00010/00100/01000/01000/01000',
 '8':'01110/10001/10001/01110/10001/10001/01110',
 '9':'01110/10001/10001/01111/00001/00001/01110',
 '+':'00000/00100/00100/11111/00100/00100/00000',
 '-':'00000/00000/00000/11111/00000/00000/00000',
 ':':'00000/00100/00100/00000/00100/00100/00000',
 '.':'00000/00000/00000/00000/00000/00100/00100',
 '_':'00000/00000/00000/00000/00000/00000/11111',
 '?':'01110/10001/00001/00010/00100/00000/00100',
 ' ':'00000/00000/00000/00000/00000/00000/00000',
}


def _text(canvas,value,x,y,scale=2):
    for char in str(value).upper():
        if x+5*scale >= canvas.shape[1]:
            break
        for row,pattern in enumerate(_FONT_ROWS.get(char,_FONT_ROWS['?']).split('/')):
            for col,pixel in enumerate(pattern):
                if pixel == '1':
                    canvas[y+row*scale:y+(row+1)*scale,x+col*scale:x+(col+1)*scale] = (25,25,25)
        x += 6*scale


def _png(board,metadata):
    image = np.full((700,620,3),(216,173,108),dtype=np.uint8)
    for i in range(19):
        p = 40+i*30
        image[40:581,p] = (60,45,25)
        image[p,40:581] = (60,45,25)
    yy,xx = np.ogrid[:700,:620]
    for r in (3,9,15):
        for c in (3,9,15):
            image[(yy-(40+r*30))**2+(xx-(40+c*30))**2 <= 9] = (40,30,20)
    for r,c in np.argwhere(board != 0):
        distance = (yy-(40+r*30))**2+(xx-(40+c*30))**2
        image[distance <= 13**2] = (25,25,25)
        image[distance <= 12**2] = (15,15,15) if board[r,c] == 1 else (245,245,238)
    _text(image,f"EPISODE {metadata.get('episode','?')}  MOVES {metadata.get('moves','?')}",22,610)
    _text(image,f"RESULT {metadata.get('result') or 'UNKNOWN'}",22,635)
    _text(image,metadata.get('skip_reason') or metadata.get('result_source') or 'COMPLETED',22,660)
    def chunk(kind,payload):
        return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
    pixels = b''.join(b'\x00'+row.tobytes() for row in image)
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',620,700,8,2,0,0,0))
            +chunk(b'IDAT',zlib.compress(pixels,6))+chunk(b'IEND',b''))


def save_game(output,episode,board,state,metadata):
    """Save PNG, JSON, and original eight-plane NPZ for EVERY completed rollout."""
    board,state = np.asarray(board).copy(),np.asarray(state,dtype=np.int16).copy()
    if board.shape != (19,19) or state.shape != (8,19,19):
        raise ValueError('expected board (19,19) and state (8,19,19)')
    metadata = dict(metadata,episode=int(episode))
    label = metadata.get('value_label')
    if label not in (None,0,1):
        raise ValueError('value_label must be 1 (Black), 0 (White), or None')
    path = Path(output)/'games'/f'episode-{episode:08d}'
    path.mkdir(parents=True,exist_ok=False)
    encoded = json.dumps(metadata,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
    (path/'result.json').write_text(encoded+'\n',encoding='utf-8')
    (path/'board.png').write_bytes(_png(board,metadata))
    states = state[None] if label is not None else np.empty((0,8,19,19),dtype=np.int16)
    results = np.asarray([label] if label is not None else [],dtype=np.int8)
    np.savez_compressed(path/'position.npz',states=states,results=results,
                        board=board,encoded_state=state,metadata=np.asarray(encoded))
    return path


def _run_id(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}',run_id):
        raise ValueError('run_id must be a simple unique filename component')
    return run_id


def upload_game(game_dir,run_id,*,api=None,data_repo=DATA_REPO):
    api = api or HfApi()
    run_id = _run_id(run_id)
    path = Path(game_dir)
    operations = [CommitOperationAdd(path_in_repo=f'runs/{run_id}/games/{path.name}/{name}',
                                    path_or_fileobj=str(path/name))
                  for name in ('board.png','result.json','position.npz')]
    return api.create_commit(repo_id=data_repo,repo_type='dataset',operations=operations,
                             commit_message=f'Add self-play {run_id}/{path.name}')


def snapshot_code(output,project_root):
    """Freeze only the execution sources for this run before any episode."""
    import shutil
    target = Path(output)/'code'
    for relative in SOURCE_FILES:
        source = Path(project_root)/relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = target/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    return target


def upload_training(output,project_root,run_id,*,api=None,model_repo=MODEL_REPO):
    """Commit local weights + allowlisted code after a completed run.

    No recursive project upload, so .env/tokens/caches/unrelated files stay local.
    Root last.pth is the next run's automatic initialization.
    """
    api = api or HfApi()
    run_id = _run_id(run_id)
    output,project_root = Path(output),Path(project_root)
    weights = output/'last.pth'
    if not weights.is_file():
        raise FileNotFoundError(weights)
    code = output/'code'
    if not code.is_dir():
        raise FileNotFoundError(f'Missing frozen source snapshot: {code}')
    operations = [CommitOperationAdd(path_in_repo='last.pth',path_or_fileobj=str(weights)),
                  CommitOperationAdd(path_in_repo=f'runs/{run_id}/last.pth',path_or_fileobj=str(weights))]
    for relative in SOURCE_FILES:
        source = code/relative
        if not source.is_file():
            raise FileNotFoundError(source)
        operations.append(CommitOperationAdd(path_in_repo=relative,path_or_fileobj=str(source)))
        operations.append(CommitOperationAdd(path_in_repo=f'runs/{run_id}/source/{relative}',path_or_fileobj=str(source)))
    initial = output/'initial.pth'
    if initial.is_file():
        operations.append(CommitOperationAdd(path_in_repo=f'runs/{run_id}/initial.pth',path_or_fileobj=str(initial)))
    for snapshot in sorted(output.glob('policy-*.pth')):
        operations.append(CommitOperationAdd(path_in_repo=f'runs/{run_id}/{snapshot.name}',path_or_fileobj=str(snapshot)))
    if (output/'run.json').is_file():
        operations.append(CommitOperationAdd(path_in_repo=f'runs/{run_id}/run.json',path_or_fileobj=str(output/'run.json')))
    return api.create_commit(repo_id=model_repo,repo_type='model',operations=operations,
                             commit_message=f'Save REINFORCE weights and code {run_id}')


def bootstrap_opponent(learner_source, *, cache_dir=None, api=None):
    """Explicit one-time baseline archive; never replaces the learner's last.pth."""
    import hashlib
    api = api or HfApi()
    info = api.model_info(FALLBACK_REPO, files_metadata=True)
    filename = 'best_policy_network_128.pth'
    if filename not in {item.rfilename for item in info.siblings}:
        raise FileNotFoundError(f'{FALLBACK_REPO}/{filename}')
    path = hf_hub_download(repo_id=FALLBACK_REPO, filename=filename,
                           revision=info.sha, repo_type='model', cache_dir=cache_dir)
    with open(path, 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    identity = f'sha256:{digest}'
    if learner_source and learner_source.get('identity') == identity:
        raise ValueError('Bootstrap model is identical to the learner; add a distinct older checkpoint first.')
    from .policy import load_policy
    validated, _ = load_policy(path, 'cpu')
    del validated
    target = f'bootstrap/{digest}/initial.pth'
    provenance = json.dumps(dict(repo_id=FALLBACK_REPO, filename=filename,
                                 revision=info.sha, sha256=digest), indent=2).encode()
    commit = api.create_commit(repo_id=MODEL_REPO, repo_type='model', operations=[
        CommitOperationAdd(path_in_repo=target, path_or_fileobj=path),
        CommitOperationAdd(path_in_repo=f'bootstrap/{digest}/source.json', path_or_fileobj=provenance),
    ], commit_message='Archive supervised initialization as first historical opponent')
    return dict(kind='huggingface', repo_id=MODEL_REPO, filename=target,
                revision=commit.oid, identity=identity)
