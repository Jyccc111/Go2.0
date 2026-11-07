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
from rl.policy import distribution, load_policy, save_checkpoint
from rl.train import rollout, reinforce_update


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

    def test_cli_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = Policy_network(size=2, num_blocks=0)
            with torch.no_grad():
                model.network[-1].bias[PASS] = 100
            source = root / 'source.pth'
            torch.save(model.network.state_dict(), source)
            source_bytes = source.read_bytes()
            from rl.train import main
            with patch('Game.Game.cal_result', side_effect=AssertionError('short game scored')):
                main(['--opponent-mode', 'self', '--checkpoint', str(source), '--output', str(root/'first'),
                  '--episodes', '1', '--max-moves', '2', '--no-upload', '--pool-size', '1',
                  '--snapshot-every', '1'])
            main(['--opponent-mode', 'self', '--checkpoint', str(root/'first'/'last.pth'), '--resume',
                  '--output', str(root/'second'), '--episodes', '1', '--max-moves', '2', '--no-upload'])
            _, checkpoint = load_policy(root/'second'/'last.pth')
            self.assertEqual(checkpoint['episode'], 2)
            for key, value in model.network.state_dict().items():
                self.assertTrue(torch.equal(value, checkpoint['model_state_dict'][key]))
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertTrue((root/'first'/'policy-1.pth').exists())
            game_dir = next((root/'first'/'games').glob('episode-*'))
            self.assertTrue((game_dir/'board.png').exists())
            self.assertIsNone(json.loads((game_dir/'result.json').read_text())['value_label'])
            with np.load(game_dir/'position.npz', allow_pickle=False) as data:
                self.assertEqual(data['states'].shape, (0,8,19,19))
                self.assertEqual(data['results'].size, 0)

    def test_cli_hub_load_archive_upload_and_retry(self):
        from rl.train import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.pth'
            torch.save(Policy_network(size=2,num_blocks=0).state_dict(), source)
            def short_resign(model, env, *args):
                state, mask = env.observe(), env.legal_mask()
                env.step(RESIGN)
                return [(state,mask,RESIGN,1)]
            with patch('rl.train.resolve_checkpoint', return_value=(str(source), {'repo':'mock'})) as load, \
                 patch('rl.train.rollout', side_effect=short_resign), \
                 patch('rl.train.upload_game') as upload, \
                 patch('rl.train.upload_training') as final_upload:
                main(['--opponent-mode', 'self', '--output',str(root/'run'),'--episodes','2'])
                self.assertIsNone(load.call_args.args[0])
                self.assertEqual(upload.call_count,2)
                final_upload.assert_called_once()
                for game_dir in (root/'run'/'games').glob('episode-*'):
                    metadata = json.loads((game_dir/'result.json').read_text())
                    self.assertEqual(metadata['result'], 'W+R')
                    self.assertEqual(metadata['value_label'], 0)
                    with np.load(game_dir/'position.npz', allow_pickle=False) as data:
                        self.assertEqual(data['states'].shape,(1,8,19,19))
                        np.testing.assert_array_equal(data['results'], [0])
                upload.reset_mock()
                final_upload.reset_mock()
                load.reset_mock()
                main(['--opponent-mode', 'self', '--output',str(root/'run'),'--upload-only'])
                load.assert_not_called()
                self.assertEqual(upload.call_count,2)
                final_upload.assert_called_once()

    def test_default_history_uses_latest_and_alternates_learner(self):
        from rl.train import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/'latest.pth'
            torch.save(Policy_network(size=2,num_blocks=0).state_dict(), checkpoint)
            latest = dict(repo_id='owner/rl', filename='last.pth', revision='new', identity='sha256:new')
            old = dict(repo_id='owner/rl', filename='last.pth', revision='old', identity='sha256:old')
            opponent = Policy_network(size=2,num_blocks=0).eval().requires_grad_(False)
            before = copy.deepcopy(opponent.state_dict())
            seen = []
            def short_game(model, env, actual_opponent, learner_color, temperature):
                self.assertIs(actual_opponent,opponent)
                seen.append(learner_color)
                # End on the learner's turn with a resignation.
                from Game import Player
                env.game.current_player = Player(learner_color)
                env._refresh()
                state,mask = env.observe(),env.legal_mask()
                env.step(RESIGN)
                return [(state,mask,RESIGN,learner_color)]
            with patch('rl.train.discover_history',return_value=(latest,[old])) as discover, \
                 patch('rl.train.download_version',return_value=str(checkpoint)) as download, \
                 patch('rl.train.HistoricalOpponentPool') as pool, \
                 patch('rl.train.rollout',side_effect=short_game):
                pool.return_value.sample.return_value=(opponent,old)
                main(['--output',str(root/'run'),'--episodes','2','--no-upload'])
                discover.assert_called_once()
                download.assert_called_once()
                self.assertEqual(download.call_args.args[0],latest)
                self.assertEqual(pool.return_value.sample.call_count,2)
            self.assertEqual(seen,[1,-1])
            for key,value in opponent.state_dict().items():
                self.assertTrue(torch.equal(value,before[key]))
            for file in (root/'run'/'games').glob('*/result.json'):
                metadata=json.loads(file.read_text())
                self.assertEqual(metadata['learner_source'],latest)
                self.assertEqual(metadata['opponent_source'],old)
            self.assertTrue((root/'run'/'initial.pth').is_file())

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
