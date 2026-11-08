"""Copy all distinct historical policies into models/ without deleting originals.

Preview: python -m rl.organize_models
Apply:   python -m rl.organize_models --apply
"""
import argparse
import json

from huggingface_hub import CommitOperationCopy, HfApi

from .hub import MODEL_REPO
from .model_store import version_details, version_path
from .opponents import _checkpoints, _identity, discover_history


def organize_models(*, api=None, apply=False, model_repo=MODEL_REPO):
    api = api or HfApi()
    head = api.model_info(repo_id=model_repo, files_metadata=True)
    learner, history = discover_history(api=api, model_repo=model_repo)
    sources = ([learner] if learner else []) + history
    # Prefer a meaningful epoch path over aliases such as last.pth. A second
    # pinned HEAD read inside discovery must not race with this migration.
    if learner and learner['revision'] != head.sha:
        raise ValueError('Repository changed during discovery; preview again')
    existing = {s.rfilename: _identity(s) for s in _checkpoints(head)}
    details = {}
    flat = {}
    for sibling in _checkpoints(head):
        identity = _identity(sibling)
        parsed = version_details(sibling.rfilename)
        if parsed:
            details[identity] = parsed[0]
            if sibling.rfilename.startswith('models/'):
                flat[identity] = sibling.rfilename
    plan, operations = [], []
    for source in sources:
        identity = source['identity']
        epoch = details.get(identity)
        old = version_details(source['filename'])
        if epoch is None and old:
            epoch = old[0]
        target = flat.get(identity) or version_path(identity.split(':', 1)[1], epoch)
        if target in existing and existing[target] != identity:
            raise ValueError(f'Content identity conflict at {target}')
        present = target in existing
        plan.append(dict(source=source, destination=target, already_present=present))
        if not present:
            operations.append(CommitOperationCopy(src_path_in_repo=source['filename'],
                              src_revision=source['revision'], path_in_repo=target))
    if apply and operations:
        # An intervening upload aborts the commit instead of mixing HEADs.
        api.create_commit(repo_id=model_repo, repo_type='model', operations=operations,
                          parent_commit=head.sha,
                          commit_message='Collect distinct policy versions in models/')
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='copy versions; preserve all originals and last.pth')
    args = parser.parse_args(argv)
    plan = organize_models(apply=args.apply)
    print(json.dumps(dict(applied=args.apply, versions=plan), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
