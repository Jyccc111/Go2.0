import copy
import subprocess
import sys
import tempfile
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from Train_net import Policy_network
from rl.game import GoAdapter, PASS, RESIGN
from unittest.mock import patch
from rl.policy import distribution, load_policy, save_checkpoint, CheckpointFileError
from rl.train import rollout, reinforce_update, action_returns


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(9)

    def test_import_has_no_app_or_training(self):
        subprocess.run([sys.executable, '-c',
            "import Reinforcement_learning; import sys; assert 'app' not in sys.modules; assert 'AI' not in sys.modules"], check=True)

    def test_board_pass_and_mask(self):
        env = GoAdapter()
        self.assertEqual(env.observe().shape, (8,19,19))
        self.assertEqual(env.observe().dtype, np.int16)
        env.step(0)  # (0,0) is a real move, not pass
        self.assertEqual(env.player, -1)
        self.assertFalse(env.legal_mask()[0])
        self.assertTrue(env.legal_mask()[RESIGN])
        with self.assertRaises(ValueError):
            env.step(0)
        env.game.lasteated[2,2] = 1
        env._refresh()
        self.assertFalse(env.legal_mask()[40])
        env.step(PASS)
        self.assertEqual(env.game.lasteated.sum(), 0)
        self.assertEqual(env.player, 1)
        env.step(PASS)
        self.assertTrue(env.done)
        with self.assertRaises(ValueError):
            env.rewards()
        self.assertFalse(env.legal_mask().any())
        with self.assertRaises(ValueError):
            env.step(PASS)

    def test_scoring_and_truncation(self):
        env = GoAdapter(komi=6.5)
        env.moves = 198  # Exercise the exact 200-action scoring boundary.
        env.step(PASS)
        env.step(PASS)
        original_board = env.game.board.board.copy()
        def original_scorer(game):
            self.assertEqual(game.komi, 6.5)
            game.board.board[0,0] = 1  # Original scorer may remove dead stones.
            return 100, 110, False
        with patch('Game.Game.cal_result', autospec=True, side_effect=original_scorer) as score:
            self.assertEqual(env.rewards(), {1:-1., -1:1.})
            env.rewards()
            score.assert_called_once()
        np.testing.assert_array_equal(env.game.board.board, original_board)
        env = GoAdapter(max_moves=2)
        env.step(0)
        env.step(1)
        self.assertTrue(env.truncated)
        with self.assertRaises(ValueError):
            env.rewards()

    def test_short_pass_games_never_score(self):
        for before in (0, 197):
            env = GoAdapter()
            env.moves = before
            env.step(PASS)
            env.step(PASS)
            with patch('Game.Game.cal_result', side_effect=AssertionError('must skip')) as score:
                self.assertEqual(env.skip_reason, 'two_passes_before_200')
                self.assertIsNone(env.result()['value_label'])
                with self.assertRaises(ValueError):
                    env.rewards()
                score.assert_not_called()

    def test_default_500_cap_skips_without_scoring(self):
        env = GoAdapter()
        self.assertEqual(env.max_moves, 500)
        env.moves = 499
        env.step(0)
        with patch('Game.Game.cal_result', side_effect=AssertionError('unfinished')):
            self.assertTrue(env.done)
            self.assertTrue(env.truncated)
            self.assertEqual(env.result()['skip_reason'], 'move_limit')
            self.assertIsNone(env.result()['value_label'])

    def test_natural_end_at_500_is_not_truncation(self):
        for last_action in (PASS, RESIGN):
            env = GoAdapter()
            env.moves = 498
            env.step(PASS)
            env.step(last_action)
            self.assertFalse(env.truncated)
            with patch('Game.Game.cal_result', return_value=(190, 171, True)) as score:
                self.assertIsNone(env.skip_reason)
                self.assertEqual(env.result()['winner'], 1)
                score.assert_called_once()

    def test_terminal_pass_bonus_is_once_and_does_not_reward_losing_pass(self):
        records = [(None, None, a, c) for a, c in
                   [(PASS, 1), (0, 1), (PASS, -1), (PASS, 1), (PASS, -1)]]
        outcome = dict(termination='two_passes', moves=300, winner=1,
                       result_source='Game.cal_result', truncated=False)
        self.assertEqual(action_returns(records, {1:1., -1:-1.}, outcome), [1,1,-1,1,-1])
        self.assertEqual(action_returns(records, {1:1., -1:-1.}, outcome, 0.05),
                         [1,1,-1,1.05,-1])
        # Same rule for White; no bonus if the winner is the frozen opponent.
        outcome['winner'] = -1
        self.assertEqual(action_returns(records, {1:-1., -1:1.}, outcome, 0.05),
                         [-1,-1,1,-1,1.05])
        only_black = [record for record in records if record[3] == 1]
        self.assertEqual(action_returns(only_black, {1:-1., -1:1.}, outcome, 0.05), [-1,-1,-1])

    def test_bonus_excludes_early_resign_truncated_and_skipped_games(self):
        records = [(None, None, PASS, 1)]
        outcome = dict(termination='two_passes', moves=300, winner=1,
                       result_source='Game.cal_result')
        for change in (dict(moves=199), dict(termination='resignation'),
                       dict(termination='move_limit'), dict(truncated=True),
                       dict(skip_reason='move_limit'), dict(result_source='resignation')):
            self.assertEqual(action_returns(records, {1:1., -1:-1.},
                                           dict(outcome, **change), 0.05), [1])
        for bonus in (-0.01, 1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                action_returns(records, {1:1., -1:-1.}, outcome, bonus)
        # Normal moves never receive a pass bonus, nor does resignation's override change.
        records = [(None,None,0,1), (None,None,RESIGN,-1)]
        rewards = {1:1., -1:-1., (-1,RESIGN):1.}
        self.assertEqual(action_returns(records,rewards,outcome,0.05), [1,1])

    def test_rollout_records_pass_probabilities_without_extra_positions(self):
        env = GoAdapter()
        model = Policy_network(size=2, num_blocks=0).eval()
        with torch.no_grad():
            model.network[-1].bias[PASS] = 100
        records = rollout(model, env)
        self.assertEqual(len(records), 2)
        self.assertEqual(env.result()['pass_counts'], {'1':1, '-1':1})
        for stats in env.pass_policy.values():
            self.assertEqual(stats['positions'], 1)
            self.assertGreater(stats['mean_probability'], 0.99)
            self.assertIsNone(stats['late_mean_probability'])
        late = GoAdapter()
        late.moves = 200
        rollout(model, late)
        self.assertTrue(all(s['late_positions'] == 1 for s in late.pass_policy.values()))

    def test_optional_pass_bonus_is_recorded_in_completed_epoch(self):
        from rl.train import main
        def end_with_passes(model, env, opponent, learner_color, temperature, sampler):
            env.moves = 198
            records = []
            for _ in range(2):
                state, mask, color = env.observe(), env.legal_mask(), env.player
                sampler.consider(state, env.game.board.board, env.moves, color)
                records.append((state, mask, PASS, color))
                env.step(PASS)
            return records
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._tiny_checkpoint(root)
            with patch('rl.train.rollout', side_effect=end_with_passes), \
                 patch('Game.Game.cal_result', return_value=(190,171,True)):
                main(['--checkpoint',str(source),'--opponent-mode','self',
                      '--output',str(root/'run'),'--epochs','1','--updates-per-epoch','1',
                      '--terminal-pass-bonus','0.05','--no-upload'])
            metadata = json.loads((root/'run'/'epochs'/'epoch-000001'/'metadata.json').read_text())
            self.assertEqual(metadata['sample_count'], 1)
            self.assertEqual(metadata['updates'], 1)
            self.assertAlmostEqual(metadata['games'][0]['terminal_pass_bonus_awarded'], 0.05)
            self.assertEqual(metadata['games'][0]['value_label'], 1)

    def test_early_resignation_uses_resignation_result(self):
        from Game import Player
        for moves in (1, 199):
            for color in (1, -1):
                env = GoAdapter(max_moves=moves if moves > 1 else 2)
                env.moves = moves - 1
                env.game.current_player = Player(color)
                env._refresh()
                state, mask = env.observe(), env.legal_mask()
                env.step(RESIGN)
                self.assertFalse(env.truncated)
                self.assertIsNone(env.skip_reason)
                with patch('Game.Game.cal_result', side_effect=AssertionError('no early scoring')) as score:
                    rewards = env.rewards()
                    result = env.result()
                    score.assert_not_called()
                self.assertEqual(rewards[color], -1)
                self.assertEqual(rewards[-color], 1)
                self.assertNotIn((color, RESIGN), rewards)
                self.assertEqual(result['winner'], -color)
                self.assertEqual(result['value_label'], int(color == -1))
                self.assertEqual(result['result_source'], 'resignation')
                model = Policy_network(size=2, num_blocks=0).eval()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                before = model.network[-1].bias[RESIGN].item()
                reinforce_update(model, optimizer, [(state,mask,RESIGN,color)], rewards)
                self.assertLess(model.network[-1].bias[RESIGN].item(), before)

    def test_resignation_reward_and_probability_direction(self):
        for color in (1, -1):
            for black_win in (True, False):
                env = GoAdapter()
                from Game import Player
                env.game.current_player = Player(color)
                env._refresh()
                env.moves = 199
                self.assertTrue(env.legal_mask()[RESIGN])
                state, mask = env.observe(), env.legal_mask()
                env.step(RESIGN)
                with patch('Game.Game.cal_result', return_value=(190,171,black_win)):
                    rewards = env.rewards()
                outcome = (1 if black_win else -1) * color
                self.assertEqual(rewards[color], outcome)
                self.assertEqual(rewards[(color, RESIGN)], -outcome)
                model = Policy_network(size=2, num_blocks=0).eval()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                before = model.network[-1].bias[RESIGN].item()
                reinforce_update(model, optimizer, [(state,mask,RESIGN,color)], rewards)
                change = model.network[-1].bias[RESIGN].item() - before
                # Correct resignation increases its weight; mistaken resignation reduces it.
                self.assertGreater(change * (-outcome), 0)

    @staticmethod
    def _tiny_checkpoint(root):
        model = Policy_network(size=2, num_blocks=0)
        with torch.no_grad():
            model.network[-1].bias[RESIGN] = 100
        path = root/'source.pth'
        torch.save(model.network.state_dict(), path)
        return path

    def test_epoch_checkpoint_and_resume_counts_zero_loss_steps(self):
        from rl.train import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=self._tiny_checkpoint(root)
            main(['--opponent-mode','self','--checkpoint',str(source),'--output',str(root/'first'),
                  '--epochs','1','--updates-per-epoch','1','--no-upload'])
            _,checkpoint=load_policy(root/'first'/'last.pth')
            self.assertEqual((checkpoint['epoch'],checkpoint['total_updates']), (1,1))
            self.assertEqual(checkpoint['config']['max_moves'], 500)
            self.assertEqual(checkpoint['config']['terminal_pass_bonus'], 0.0)
            main(['--opponent-mode','self','--checkpoint',str(root/'first'/'last.pth'),'--resume',
                  '--output',str(root/'second'),'--epochs','1','--updates-per-epoch','1','--no-upload'])
            _,checkpoint=load_policy(root/'second'/'last.pth')
            self.assertEqual((checkpoint['epoch'],checkpoint['total_updates'],checkpoint['episode']), (2,2,2))
            self.assertTrue((root/'second'/'epochs'/'epoch-000002'/'ready.json').is_file())
            self.assertEqual((root/'second'/'last.pth').read_bytes(),
                             (root/'second'/'epochs'/'epoch-000002'/'policy.pth').read_bytes())

    def test_epochs_exact_updates_skip_no_actions_and_historical_provenance(self):
        from rl.train import main
        import rl.train as train
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=self._tiny_checkpoint(root)
            latest=dict(repo_id='owner/rl',filename='last.pth',revision='new',identity='sha256:new')
            old=dict(repo_id='owner/rl',filename='last.pth',revision='old',identity='sha256:old')
            opponent=Policy_network(size=2,num_blocks=0).eval().requires_grad_(False)
            before=copy.deepcopy(opponent.state_dict())
            kinds=iter(['skip','no_actions','update','update','update','update'])
            colors=[]
            def game(model,env,actual_opponent,learner_color,temperature,sampler):
                self.assertIs(actual_opponent,opponent)
                colors.append(learner_color)
                state,mask=env.observe(),env.legal_mask()
                sampler.consider(state,env.game.board.board,env.moves,env.player)
                kind=next(kinds)
                if kind=='skip':
                    env.step(PASS)
                    env.step(PASS)
                    return []
                if kind=='no_actions':
                    env.step(RESIGN)
                    return []
                if env.player != learner_color:
                    env.step(PASS)
                    state,mask=env.observe(),env.legal_mask()
                    sampler.consider(state,env.game.board.board,env.moves,env.player)
                env.step(RESIGN)
                return [(state,mask,RESIGN,learner_color)]
            with patch('rl.train.discover_history',return_value=(latest,[old])) as discover, \
                 patch('rl.train.download_version',return_value=str(source)), \
                 patch('rl.train.HistoricalOpponentPool') as pool, \
                 patch('rl.train.rollout',side_effect=game), \
                 patch('rl.train.save_checkpoint',wraps=train.save_checkpoint) as save, \
                 patch('rl.train.upload_epoch_value') as upload_value, \
                 patch('rl.train.upload_training') as upload_model:
                pool.return_value.sample.return_value=(opponent,old)
                main(['--output',str(root/'run'),'--epochs','2','--updates-per-epoch','2'])
                discover.assert_called_once()
                self.assertEqual(save.call_count,2)
                self.assertEqual(upload_value.call_count,2)
                self.assertEqual(upload_model.call_count,2)
                self.assertEqual([call.kwargs['epoch'] for call in upload_model.call_args_list],[1,2])
            self.assertEqual(colors,[1,-1,1,-1,1,-1])
            for key,value in opponent.state_dict().items():
                self.assertTrue(torch.equal(value,before[key]))
            manifests=[]
            for epoch in (1,2):
                path=root/'run'/'epochs'/f'epoch-{epoch:06d}'
                metadata=json.loads((path/'metadata.json').read_text())
                self.assertEqual(metadata['updates'],2)
                manifests.append(metadata)
                with np.load(path/'value.npz',allow_pickle=False) as data:
                    self.assertEqual(len(set(data['episodes'])),len(data['states']))
                with __import__('zipfile').ZipFile(path/'boards.zip') as archive:
                    self.assertEqual(len(archive.namelist()),metadata['attempted_games']*2)
            self.assertEqual([m['attempted_games'] for m in manifests],[4,2])
            self.assertEqual([m['sample_count'] for m in manifests],[3,2])
            _,checkpoint=load_policy(root/'run'/'last.pth')
            self.assertEqual((checkpoint['epoch'],checkpoint['total_updates'],checkpoint['episode']),(2,4,6))
            for file in (root/'run'/'games').glob('*/result.json'):
                metadata=json.loads(file.read_text())
                self.assertEqual(metadata['learner_source'],latest)
                self.assertEqual(metadata['opponent_source'],old)
                self.assertEqual(metadata['board_kind'],'random_pre_action')

    def test_partial_epoch_is_not_saved_or_uploaded_as_complete(self):
        from rl.train import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            model=Policy_network(size=2,num_blocks=0)
            with torch.no_grad():
                model.network[-1].bias[PASS]=100
            source=root/'source.pth'
            torch.save(model.state_dict(),source)
            with patch('rl.train.upload_epoch_value') as value,patch('rl.train.upload_training') as upload, \
                 self.assertRaisesRegex(RuntimeError,'incomplete'):
                main(['--checkpoint',str(source),'--opponent-mode','self','--output',str(root/'run'),
                      '--updates-per-epoch','1','--max-games-per-epoch','2'])
            value.assert_not_called()
            upload.assert_not_called()
            path=root/'run'/'epochs'/'epoch-000001'
            metadata=json.loads((path/'metadata.json').read_text())
            self.assertFalse(metadata['complete'])
            self.assertEqual(metadata['updates'],0)
            self.assertFalse((path/'ready.json').exists())
            self.assertFalse((root/'run'/'last.pth').exists())

    def test_epoch_upload_failure_retries_only_failed_repository(self):
        from rl.train import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=self._tiny_checkpoint(root)
            with patch('rl.train.upload_epoch_value') as value, \
                 patch('rl.train.upload_training',side_effect=[ConnectionError('offline'),None]) as model:
                with self.assertRaises(ConnectionError):
                    main(['--checkpoint',str(source),'--opponent-mode','self','--output',str(root/'run'),
                          '--updates-per-epoch','1'])
                path=root/'run'/'epochs'/'epoch-000001'
                self.assertTrue(json.loads((path/'metadata.json').read_text())['complete'])
                self.assertTrue(json.loads((path/'uploads.json').read_text())['value'])
                main(['--output',str(root/'run'),'--upload-only'])
                self.assertEqual(value.call_count,1)
                self.assertEqual(model.call_count,2)
                main(['--output',str(root/'run'),'--upload-only'])
                self.assertEqual(model.call_count,2)
                ready=json.loads((path/'ready.json').read_text())
                ready['epoch']=2
                (path/'ready.json').write_text(json.dumps(ready))
                with self.assertRaisesRegex(ValueError,'pairing'):
                    main(['--output',str(root/'run'),'--upload-only'])

    def test_no_records_does_not_step_optimizer(self):
        model=Policy_network(size=2,num_blocks=0)
        optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
        with patch.object(optimizer,'step') as step:
            self.assertIsNone(reinforce_update(model,optimizer,[],{1:1,-1:-1}))
            step.assert_not_called()

    def test_pointer_diagnostics_and_legacy_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pth'
            for data,message in ((b'../../blobs/hash','symlink target'),
                                 (b'version https://git-lfs.github.com/spec/v1','Git LFS'),
                                 (b'', 'Empty checkpoint')):
                path.write_bytes(data)
                with self.assertRaisesRegex(CheckpointFileError,message):
                    load_policy(path)
            model=Policy_network(size=2,num_blocks=0)
            torch.save(model.state_dict(),path,_use_new_zipfile_serialization=False)
            loaded,_=load_policy(path)
            self.assertEqual(loaded.size,2)

    def test_masks_and_gradient_sign(self):
        mask = np.zeros(363, bool)
        mask[[0,PASS]] = True
        logits = torch.zeros(363, requires_grad=True)
        dist = distribution(logits, mask)
        self.assertEqual(dist.probs[362].item(), 0)
        self.assertEqual(dist.probs[1].item(), 0)
        (-dist.log_prob(torch.tensor(0))).backward()
        self.assertLess(logits.grad[0].item(), 0)
        logits.grad.zero_()
        distribution(logits, mask).log_prob(torch.tensor(0)).backward()
        self.assertGreater(logits.grad[0].item(), 0)

    def test_rollout_update_opponent_and_checkpoints(self):
        model = Policy_network(size=2, num_blocks=0)
        # Force short completed games while retaining a nonzero policy gradient.
        with torch.no_grad():
            model.network[-1].bias[PASS] = 12
        opponent = copy.deepcopy(model)
        before_opponent = copy.deepcopy(opponent.state_dict())
        env = GoAdapter()
        records = rollout(model, env, opponent, learner_color=-1)
        self.assertTrue(records)
        self.assertTrue(all(r[3] == -1 for r in records))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.network[-1].bias.detach().clone()
        loss = reinforce_update(model, optimizer, records, {1:-1., -1:1.})
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.network[-1].bias))
        for key, value in opponent.state_dict().items():
            self.assertTrue(torch.equal(value, before_opponent[key]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pth'
            save_checkpoint(path, model, optimizer, 2, {})
            loaded, checkpoint = load_policy(path)
            self.assertEqual(checkpoint['episode'], 2)
            for key,value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]))
            # Raw outer wrapper and raw Sequential formats, including blocks.
            residual = Policy_network(size=3, num_blocks=1)
            for state in (residual.state_dict(), residual.network.state_dict()):
                torch.save(state, path)
                loaded, _ = load_policy(path)
                self.assertEqual((loaded.size, loaded.num_blocks), (3,1))
            self.assertFalse(list(Path(directory).glob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
