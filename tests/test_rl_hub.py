import json
import hashlib
from types import SimpleNamespace as NS
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch
import zlib
import numpy as np
from rl.hub import (MODEL_REPO,FALLBACK_REPO,DATA_REPO,SOURCE_FILES,
                    resolve_checkpoint,save_game,upload_game,snapshot_code,upload_training,bootstrap_opponent)


class HubSmokeTests(unittest.TestCase):
    def test_rl_model_preferred_and_cache_passed(self):
        api = Mock()
        api.list_repo_files.return_value = ['best_policy_network_128.pth','last.pth']
        with patch('rl.hub.hf_hub_download',return_value='/cached/last.pth') as download:
            path,source = resolve_checkpoint(api=api,cache_dir='/local/cache')
        self.assertEqual(path,'/cached/last.pth')
        self.assertEqual(source['repo_id'],MODEL_REPO)
        self.assertEqual(source['filename'],'last.pth')
        api.list_repo_files.assert_called_once_with(repo_id=MODEL_REPO,repo_type='model')
        download.assert_called_once_with(repo_id=MODEL_REPO,filename='last.pth',repo_type='model',cache_dir='/local/cache')

    def test_only_empty_rl_listing_triggers_exact_fallback(self):
        api = Mock()
        api.list_repo_files.side_effect = [['.gitattributes'],['best_policy_network_128.pth']]
        with patch('rl.hub.hf_hub_download',return_value='/cached/best.pth'):
            _,source = resolve_checkpoint(api=api)
        self.assertEqual(source['repo_id'],FALLBACK_REPO)
        self.assertEqual(source['filename'],'best_policy_network_128.pth')
        api = Mock()
        api.list_repo_files.side_effect = PermissionError('private repo')
        with self.assertRaises(PermissionError),patch('rl.hub.hf_hub_download') as download:
            resolve_checkpoint(api=api)
        download.assert_not_called()
        self.assertEqual(api.list_repo_files.call_count,1)
        api = Mock()
        api.list_repo_files.side_effect = ConnectionError('offline')
        with self.assertRaises(ConnectionError):
            resolve_checkpoint(api=api)
        self.assertEqual(api.list_repo_files.call_count,1)

    def test_local_checkpoint_never_uses_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'local.pth'
            path.write_bytes(b'test')
            api = Mock()
            result,source = resolve_checkpoint(path,api=api)
            self.assertEqual(Path(result),path.resolve())
            self.assertEqual(source['kind'],'local')
            api.list_repo_files.assert_not_called()

    def test_per_game_artifacts_labels_and_upload(self):
        board = np.zeros((19,19),dtype=np.int8)
        board[0,0],board[18,18] = 1,-1
        state = np.zeros((8,19,19),dtype=np.int16)
        with tempfile.TemporaryDirectory() as directory:
            path = save_game(directory,1,board,state,dict(moves=3,result='W+R',value_label=0,result_source='resignation'))
            self.assertEqual(json.loads((path/'result.json').read_text())['result'],'W+R')
            with np.load(path/'position.npz',allow_pickle=False) as data:
                self.assertEqual(data['states'].shape,(1,8,19,19))
                self.assertEqual(data['states'].dtype,np.int16)
                self.assertEqual(data['results'].tolist(),[0])
                np.testing.assert_array_equal(data['board'],board)
            png = (path/'board.png').read_bytes()
            self.assertEqual(png[:8],b'\x89PNG\r\n\x1a\n')
            self.assertEqual(struct.unpack('>II',png[16:24]),(620,700))
            cursor,compressed = 8,bytearray()
            while cursor < len(png):
                length = struct.unpack('>I',png[cursor:cursor+4])[0]
                kind = png[cursor+4:cursor+8]
                payload = png[cursor+8:cursor+8+length]
                crc = struct.unpack('>I',png[cursor+8+length:cursor+12+length])[0]
                self.assertEqual(crc,zlib.crc32(kind+payload)&0xffffffff)
                if kind == b'IDAT':
                    compressed.extend(payload)
                cursor += 12+length
            self.assertEqual(len(zlib.decompress(compressed)),700*(620*3+1))
            api = Mock()
            upload_game(path,'run-123',api=api)
            kwargs = api.create_commit.call_args.kwargs
            self.assertEqual(kwargs['repo_id'],DATA_REPO)
            self.assertEqual(kwargs['repo_type'],'dataset')
            self.assertEqual({op.path_in_repo for op in kwargs['operations']},
                             {f'runs/run-123/games/{path.name}/{name}' for name in ('board.png','result.json','position.npz')})
            skipped = save_game(directory,2,board,state,dict(moves=2,result='UNKNOWN',value_label=None,skip_reason='early_pass'))
            with np.load(skipped/'position.npz',allow_pickle=False) as data:
                self.assertEqual(data['states'].shape,(0,8,19,19))
                self.assertEqual(data['results'].shape,(0,))
                self.assertEqual(data['encoded_state'].shape,(8,19,19))

    def test_upload_failure_preserves_local_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_game(directory,1,np.zeros((19,19)),np.zeros((8,19,19)),dict(value_label=None))
            api = Mock()
            api.create_commit.side_effect = ConnectionError('offline')
            with self.assertRaises(ConnectionError):
                upload_game(path,'run-123',api=api)
            self.assertEqual(len(list(path.iterdir())),3)

    def test_explicit_bootstrap_archives_baseline_without_replacing_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            initial=Path(directory)/'initial.pth'
            initial.write_bytes(b'supervised baseline')
            digest=hashlib.sha256(initial.read_bytes()).hexdigest()
            api=Mock()
            api.model_info.return_value=NS(sha='supervised-commit',siblings=[NS(rfilename='best_policy_network_128.pth')])
            api.create_commit.return_value=NS(oid='bootstrap-commit')
            with patch('rl.hub.hf_hub_download',return_value=str(initial)) as download, \
                 patch('rl.policy.load_policy',return_value=(None,{})):
                source=bootstrap_opponent(dict(identity='sha256:latest'),api=api)
            self.assertEqual(download.call_args.kwargs['revision'],'supervised-commit')
            self.assertEqual(source['revision'],'bootstrap-commit')
            self.assertEqual(source['identity'],'sha256:'+digest)
            paths={op.path_in_repo for op in api.create_commit.call_args.kwargs['operations']}
            self.assertNotIn('last.pth',paths)
            self.assertIn('bootstrap/'+digest+'/initial.pth',paths)
            api.create_commit.reset_mock()
            with patch('rl.hub.hf_hub_download',return_value=str(initial)), self.assertRaises(ValueError):
                bootstrap_opponent(dict(identity='sha256:'+digest),api=api)
            api.create_commit.assert_not_called()

    def test_invalid_bootstrap_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'broken.pth'
            path.write_bytes(b'invalid weights')
            api=Mock()
            api.model_info.return_value=NS(sha='seed',siblings=[NS(rfilename='best_policy_network_128.pth')])
            with patch('rl.hub.hf_hub_download',return_value=str(path)), \
                 patch('rl.policy.load_policy',side_effect=ValueError('incompatible model')), \
                 self.assertRaises(ValueError):
                bootstrap_opponent(dict(identity='sha256:latest'),api=api)
            api.create_commit.assert_not_called()

    def test_training_upload_uses_frozen_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project,output = root/'project',root/'run'
            for relative in SOURCE_FILES:
                source = project/relative
                source.parent.mkdir(parents=True,exist_ok=True)
                source.write_text('original code')
            (project/'.env').write_text('secret')
            output.mkdir()
            snapshot_code(output,project)
            (project/'Game.py').write_text('changed after training')
            (output/'last.pth').write_bytes(b'weights')
            (output/'initial.pth').write_bytes(b'initial weights')
            (output/'policy-20.pth').write_bytes(b'snapshot')
            (output/'run.json').write_text('{}')
            api = Mock()
            upload_training(output,project,'run-123',api=api)
            kwargs = api.create_commit.call_args.kwargs
            paths = {op.path_in_repo for op in kwargs['operations']}
            self.assertEqual(kwargs['repo_id'],MODEL_REPO)
            self.assertEqual(kwargs['repo_type'],'model')
            self.assertIn('last.pth',paths)
            self.assertIn('runs/run-123/last.pth',paths)
            self.assertIn('runs/run-123/initial.pth',paths)
            self.assertIn('runs/run-123/policy-20.pth',paths)
            self.assertIn('runs/run-123/run.json',paths)
            self.assertNotIn('.env',paths)
            operation = next(op for op in kwargs['operations'] if op.path_in_repo == 'Game.py')
            self.assertEqual(Path(operation.path_or_fileobj).read_text(),'original code')
            epoch=output/'epochs'/'epoch-000001'
            epoch.mkdir(parents=True)
            (epoch/'policy.pth').write_bytes(b'epoch one immutable model')
            upload_training(output,project,'run-123',api=api,epoch=1)
            operations=api.create_commit.call_args.kwargs['operations']
            current=next(op for op in operations if op.path_in_repo=='last.pth')
            self.assertEqual(Path(current.path_or_fileobj).read_bytes(),b'epoch one immutable model')
            self.assertIn('runs/run-123/epochs/epoch-000001/policy.pth',
                          {op.path_in_repo for op in operations})



if __name__ == '__main__':
    unittest.main()
