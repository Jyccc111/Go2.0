"""Flat policy naming and nondestructive server-side historical organization."""
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from huggingface_hub import CommitOperationCopy
from rl.model_store import version_path, version_details, local_version_path
from rl.organize_models import organize_models


def file(name, digest):
    return NS(rfilename=name, lfs=NS(sha256=digest), blob_id=None)


class ModelStoreTests(unittest.TestCase):
    def test_content_names_cannot_collide_between_runs_at_same_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.pth'
            path.write_bytes(b'first run')
            first = local_version_path(path, 5)
            path.write_bytes(b'another run')
            second = local_version_path(path, 5)
        self.assertNotEqual(first, second)
        self.assertEqual(Path(first).parent.as_posix(), 'models')
        self.assertEqual(version_details(first), (5, hashlib.sha256(b'first run').hexdigest()))
        self.assertIsNone(version_details('../models/epoch-5.pth'))
        with self.assertRaises(ValueError):
            version_path('../escape')

    def test_preview_then_apply_copies_each_identity_and_preserves_latest(self):
        a, b = 'a' * 64, 'b' * 64
        oldpath = 'runs/old/epochs/epoch-000005/policy.pth'
        head = NS(sha='head', siblings=[file('last.pth', a), file(oldpath, a)])
        learner = dict(identity='sha256:' + a, filename='last.pth', revision='head')
        old = dict(identity='sha256:' + b, filename='last.pth', revision='ancestor')
        api = Mock(); api.model_info.return_value = head
        with patch('rl.organize_models.discover_history', return_value=(learner, [old])):
            preview = organize_models(api=api)
            api.create_commit.assert_not_called()
            applied = organize_models(api=api, apply=True)
        self.assertEqual(applied, preview)
        self.assertEqual(len(applied), 2)
        operations = api.create_commit.call_args.kwargs['operations']
        self.assertTrue(all(isinstance(op, CommitOperationCopy) for op in operations))
        self.assertEqual({op.path_in_repo for op in operations},
                         {version_path(a, 5), version_path(b)})
        self.assertEqual(operations[1].src_revision, 'ancestor')
        self.assertEqual(api.create_commit.call_args.kwargs['parent_commit'], 'head')

    def test_repeated_organization_does_not_commit_or_duplicate(self):
        digest = 'a' * 64
        path = version_path(digest, 5)
        head = NS(sha='head', siblings=[file(path, digest), file('last.pth', digest)])
        source = dict(identity='sha256:' + digest, filename='last.pth', revision='head')
        api = Mock();api.model_info.return_value = head
        with patch('rl.organize_models.discover_history', return_value=(source, [])):
            plan = organize_models(api=api, apply=True)
        self.assertTrue(plan[0]['already_present'])
        api.create_commit.assert_not_called()

    def test_changed_head_and_content_conflicts_abort(self):
        digest = 'a' * 64
        source = dict(identity='sha256:' + digest, filename='last.pth', revision='other')
        api = Mock();api.model_info.return_value = NS(sha='head', siblings=[])
        with patch('rl.organize_models.discover_history', return_value=(source, [])):
            with self.assertRaises(ValueError):
                organize_models(api=api, apply=True)
        api.create_commit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
