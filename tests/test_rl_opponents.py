"""Historical opponent metadata and lazy loading, with no live Hub requests."""
from types import SimpleNamespace as NS
from pathlib import Path
import tempfile
import hashlib
import random
import unittest
from unittest.mock import Mock, patch

from rl.opponents import (discover_history, download_version,
                          HistoricalOpponentPool, NoHistoricalOpponentError)


def weight(name, identity, *, lfs=True, dictionary=False):
    info = {'sha256': identity} if dictionary else NS(sha256=identity)
    return NS(rfilename=name, lfs=info if lfs else None, blob_id=identity)


def metadata(revision, *files):
    return NS(sha=revision, siblings=list(files))


def source(identity='old', revision='old-commit', filename='last.pth'):
    return dict(kind='huggingface', repo_id='owner/policy', revision=revision,
                filename=filename, identity='sha256:'+identity)


class HistoricalOpponentTests(unittest.TestCase):
    def test_pins_latest_and_collects_ancestors_and_head_snapshots(self):
        api = Mock()
        head = metadata('head', weight('last.pth','new'),
                        weight('runs/new/last.pth','new'),
                        weight('runs/old/last.pth','old1'),
                        weight('bootstrap/seed/initial.pth','seed'))
        old = metadata('commit-old', weight('last.pth','old2'),
                       weight('runs/old/last.pth','old1'))
        empty = metadata('initial')
        api.model_info.side_effect = [head, old, empty]
        api.list_repo_commits.return_value = [NS(commit_id='head'),
            NS(commit_id='commit-old'), NS(commit_id='initial')]
        learner, candidates = discover_history(api=api, model_repo='owner/policy')
        self.assertEqual(learner['filename'], 'last.pth')
        self.assertEqual(learner['revision'], 'head')
        self.assertEqual(learner['identity'], 'sha256:new')
        self.assertEqual({c['identity'] for c in candidates},
                         {'sha256:old1','sha256:old2','sha256:seed'})
        self.assertEqual(len(candidates), 3)
        self.assertTrue(all(c['kind']=='huggingface' for c in candidates))
        self.assertEqual(next(c for c in candidates if c['identity']=='sha256:old2')['revision'], 'commit-old')
        api.list_repo_commits.assert_called_once_with(
            repo_id='owner/policy', repo_type='model', revision='head')
        api.model_info.assert_any_call(repo_id='owner/policy', revision='commit-old', files_metadata=True)

    def test_current_root_and_run_copy_are_not_old_versions(self):
        api = Mock()
        api.model_info.side_effect = [metadata('head',weight('last.pth','new'),
             weight('runs/run/last.pth','new')), metadata('initial')]
        api.list_repo_commits.return_value = [NS(commit_id='head'),NS(commit_id='initial')]
        _, candidates = discover_history(api=api)
        self.assertEqual(candidates, [])
        with self.assertRaisesRegex(NoHistoricalOpponentError, '--bootstrap-opponent'):
            HistoricalOpponentPool(candidates)

    def test_same_model_in_multiple_commits_is_one_uniform_candidate(self):
        api = Mock()
        api.model_info.side_effect = [metadata('head',weight('last.pth','new')),
            metadata('old1',weight('last.pth','old')),
            metadata('old2',weight('last.pth','old'))]
        api.list_repo_commits.return_value = [NS(commit_id='head'),
            NS(commit_id='old1'), NS(commit_id='old2')]
        _, candidates = discover_history(api=api)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['revision'], 'old1')

    def test_blob_fallback_and_lfs_dictionary_metadata(self):
        api = Mock()
        api.model_info.return_value = metadata('head',weight('last.pth','new',dictionary=True),
            weight('runs/old/last.pth','old',lfs=False))
        api.list_repo_commits.return_value = [NS(commit_id='head')]
        learner,candidates = discover_history(api=api)
        self.assertEqual(learner['identity'],'sha256:new')
        self.assertEqual(candidates[0]['identity'],'blob:old')

    def test_missing_identity_fails_instead_of_using_duplicate(self):
        api = Mock()
        api.model_info.return_value = metadata('head',NS(rfilename='last.pth',lfs=None,blob_id=None))
        with self.assertRaisesRegex(ValueError,'Missing content identity'):
            discover_history(api=api)

    def test_accessible_empty_repository_does_not_mask_network_errors(self):
        api = Mock()
        api.model_info.return_value = metadata('empty')
        api.list_repo_commits.return_value = [NS(commit_id='empty')]
        self.assertEqual(discover_history(api=api), (None,[]))
        for error in (PermissionError('private'), ConnectionError('offline')):
            api = Mock()
            api.model_info.side_effect = error
            with self.assertRaises(type(error)):
                discover_history(api=api)
            api.list_repo_commits.assert_not_called()

    def test_no_main_revision_only_counts_empty_when_repo_is_accessible(self):
        class NoMain(Exception):
            pass
        api = Mock()
        api.model_info.side_effect = NoMain('no main')
        api.repo_info.return_value = NS(sha=None)
        with patch('rl.opponents.RevisionNotFoundError',NoMain):
            self.assertEqual(discover_history(api=api),(None,[]))
            api.repo_info.side_effect = PermissionError('private')
            with self.assertRaises(PermissionError):
                discover_history(api=api)

    def test_download_always_uses_pinned_revision(self):
        item=source()
        with patch('rl.opponents.hf_hub_download',return_value='/cached/model.pth') as download, \
             patch('rl.opponents.verify_checkpoint_file'):
            self.assertEqual(download_version(item,'/cache'),'/cached/model.pth')
        download.assert_called_once_with(repo_id='owner/policy',filename='last.pth',
            revision='old-commit',repo_type='model',cache_dir='/cache')
        with self.assertRaisesRegex(ValueError,'pinned'):
            download_version(dict(item,revision=None))

    def test_checksum_failure_retries_same_revision_once(self):
        from rl.policy import CheckpointFileError
        with tempfile.TemporaryDirectory() as directory:
            wrong=Path(directory)/'wrong.pth'
            good=Path(directory)/'good.pth'
            wrong.write_bytes(b'../../blobs/pointer')
            good.write_bytes(b'actual checkpoint bytes')
            version=dict(source(),identity='sha256:'+hashlib.sha256(good.read_bytes()).hexdigest())
            with patch('rl.opponents.hf_hub_download',side_effect=[str(wrong),str(good)]) as download:
                self.assertEqual(download_version(version),str(good))
                self.assertEqual(download.call_count,2)
                self.assertTrue(download.call_args.kwargs['force_download'])
                self.assertEqual(download.call_args.kwargs['revision'],version['revision'])
            with patch('rl.opponents.hf_hub_download',return_value=str(wrong)) as download, \
                 self.assertRaises(CheckpointFileError):
                download_version(version)
            self.assertEqual(download.call_count,2)

    def test_pool_loads_only_uniform_choice_and_freezes_opponent(self):
        candidates=[source('first','commit1'),source('second','commit2')]
        pool=HistoricalOpponentPool(candidates,cache_dir='/cache')
        chosen=Mock()
        chosen.choice.return_value=candidates[1]
        model=Mock()
        with patch('rl.opponents.download_version',return_value='/cached/chosen.pth') as download, \
             patch('rl.opponents.load_policy',return_value=(model,{})) as load:
            result,provenance=pool.sample(device='cuda',rng=chosen)
        self.assertIs(result,model)
        self.assertEqual(provenance,candidates[1])
        chosen.choice.assert_called_once_with(pool.candidates)
        download.assert_called_once_with(candidates[1],cache_dir='/cache')
        load.assert_called_once_with('/cached/chosen.pth','cuda')
        model.eval.assert_called_once()
        model.requires_grad_.assert_called_once_with(False)
        self.assertEqual(len(HistoricalOpponentPool(candidates+candidates).candidates),2)


if __name__=='__main__':
    unittest.main()
