"""Uniform intermediate-position sampling and epoch value export, no live I/O."""
from itertools import product
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from rl.value_data import ReservoirPositionSampler, EpochValueDataset, upload_epoch_value


def position(index, player=1):
    return dict(state=np.full((8,19,19),index,dtype=np.int16),
                board=np.zeros((19,19),dtype=np.int8), move_index=index, player=player)


class ScriptedRng:
    def __init__(self, choices):
        self.choices = iter(choices)

    def randrange(self, stop):
        result = next(self.choices)
        assert 0 <= result < stop
        return result


class EpochValueDataTests(unittest.TestCase):
    def test_reservoir_is_uniform_by_exact_small_enumeration(self):
        chosen = [0,0,0]
        # All six equally likely reservoir paths over three offered positions.
        for second,third in product(range(2),range(3)):
            sampler=ReservoirPositionSampler(ScriptedRng([0,second,third]))
            for index in (1,2,3):
                sampler.consider(**position(index, 1 if index%2 else -1))
            chosen[sampler.sample['move_index']-1] += 1
            self.assertEqual(sampler.seen,3)
        self.assertEqual(chosen,[2,2,2])

    def test_sampler_copies_and_accepts_initial_position(self):
        sampler=ReservoirPositionSampler(random.Random(4))
        self.assertIsNone(sampler.sample)
        candidate=position(1)
        sampler.consider(**candidate)
        candidate['state'].fill(99)
        candidate['board'].fill(-1)
        self.assertEqual(int(sampler.sample['state'][0,0,0]),1)
        self.assertEqual(int(sampler.sample['board'][0,0]),0)
        first=ReservoirPositionSampler(random.Random(3))
        first.consider(**dict(position(1),move_index=0))
        self.assertEqual(first.sample['move_index'],0)
        with self.assertRaises(ValueError):
            sampler.consider(**dict(position(1),move_index=-1))

    def test_one_sample_per_game_correct_binary_labels_and_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=EpochValueDataset(directory,'run')
            sample=position(4,-1)
            self.assertTrue(writer.add_game(1,sample,dict(value_label=1,skip_reason=None,history=[{'large':'discarded'}])))
            sample['state'].fill(99)
            self.assertTrue(writer.add_game(2,position(5),dict(value_label=0)))
            self.assertFalse(writer.add_game(3,position(3),dict(value_label=None,skip_reason='early_pass')))
            self.assertFalse(writer.add_game(4,position(2),dict(value_label=None)))
            self.assertFalse(writer.add_game(5,None,dict(value_label=0)))
            with self.assertRaises(ValueError):
                writer.add_game(1,position(1),dict(value_label=1))
            with self.assertRaises(ValueError):
                writer.add_game(6,position(1),dict(value_label=2))
            path=writer.save(epoch=1,updates=2)
            metadata=json.loads((path/'metadata.json').read_text())
            self.assertEqual(metadata['attempted_games'],5)
            self.assertEqual(metadata['sample_count'],2)
            self.assertEqual(metadata['updates'],2)
            self.assertEqual(metadata['skipped_games'],1)
            self.assertEqual(metadata['unlabelled_games'],1)
            self.assertEqual(metadata['no_eligible_position_games'],1)
            self.assertNotIn('history',metadata['games'][0])
            with np.load(path/'value.npz',allow_pickle=False) as data:
                from Train_value import validate_value_arrays
                validate_value_arrays(data['states'],data['results'])
                self.assertEqual(data['states'].shape,(2,8,19,19))
                self.assertEqual(data['states'].dtype,np.int16)
                np.testing.assert_array_equal(data['results'],[1,0])
                np.testing.assert_array_equal(data['episodes'],[1,2])
                np.testing.assert_array_equal(data['move_indices'],[4,5])
                np.testing.assert_array_equal(data['players'],[-1,1])
                self.assertEqual(int(data['states'][0,0,0,0]),4)
                self.assertEqual(json.loads(str(data['metadata']))['sample_count'],2)
            self.assertEqual(set(p.name for p in path.iterdir()),{'metadata.json','value.npz'})

    def test_empty_shapes_and_no_empty_dataset_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=EpochValueDataset(directory)
            path=writer.save(1,0)
            with np.load(path/'value.npz',allow_pickle=False) as data:
                self.assertEqual(data['states'].shape,(0,8,19,19))
                self.assertEqual(data['boards'].shape,(0,19,19))
                self.assertEqual(data['results'].shape,(0,))
            api=Mock()
            self.assertIsNone(upload_epoch_value(path,'run',api=api))
            api.create_commit.assert_not_called()

    def test_one_atomic_dataset_commit_per_complete_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=EpochValueDataset(directory)
            writer.add_game(9,position(3),dict(value_label=0))
            path=writer.save(2,4096)
            api=Mock()
            upload_epoch_value(path,'run-123',api=api)
            api.create_commit.assert_called_once()
            kwargs=api.create_commit.call_args.kwargs
            self.assertEqual(kwargs['repo_id'],'Jycccc111/Go_value_data')
            self.assertEqual(kwargs['repo_type'],'dataset')
            self.assertEqual({op.path_in_repo for op in kwargs['operations']},{
                'runs/run-123/epochs/epoch-000002/value.npz',
                'runs/run-123/epochs/epoch-000002/metadata.json'})
            (path/'value.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                upload_epoch_value(path,'run-123',api=api)
            self.assertEqual(api.create_commit.call_count,1)

    def test_incomplete_epoch_is_local_only_and_upload_failure_keeps_files(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=EpochValueDataset(directory)
            writer.add_game(1,position(2),dict(value_label=1))
            path=writer.save(1,1,complete=False)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                upload_epoch_value(path,'run')
            path=writer.save(1,2,complete=True)
            api=Mock()
            api.create_commit.side_effect=ConnectionError('offline')
            with self.assertRaises(ConnectionError):
                upload_epoch_value(path,'run',api=api)
            self.assertTrue((path/'value.npz').is_file())
            self.assertTrue((path/'metadata.json').is_file())


if __name__=='__main__':
    unittest.main()
