"""Independent web policy choices and resignation; no live model downloads."""
import copy
import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import torch

WEB_DEPENDENCIES = all(importlib.util.find_spec(name) for name in ('flask', 'sgfmill'))
if WEB_DEPENDENCIES:
    import AI as ai_module
    import app as web
from Game import Game, Player
from GameSession import GameSession, Type_player
from Train_net import Policy_network


class FakeAI:
    def __init__(self, policy_model, action=None):
        self.policy_model = policy_model
        self.predict_action = Mock(return_value=action or {'kind': 'pass'})
        row = 3 if policy_model == 'supervised' else 7
        self.recommend_moves = Mock(return_value=[{'row': row, 'col': row, 'score': 1.0}])


@unittest.skipUnless(WEB_DEPENDENCIES, 'Web tests require flask and sgfmill')
class WebPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.directory.name) / 'policy.pth'
        model = Policy_network(size=2, num_blocks=1)
        torch.save({'model_state_dict': model.network.state_dict()}, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.session = GameSession()
        self.globals = patch.multiple(
            web, game_session=self.session, game=self.session.game,
            recommendation_ai={}, sgf_replay=None,
        )
        self.globals.start()
        self.addCleanup(self.globals.stop)
        self.config = patch.dict(web.app.config, TESTING=True)
        self.config.start()
        self.addCleanup(self.config.stop)
        self.client = web.app.test_client()

    def test_ai_loads_selected_repo_and_existing_architecture(self):
        for selection, repo, filename in (
            (None, 'Jycccc111/Go-Policy', 'best_policy_network_128.pth'),
            ('rl', 'Jycccc111/Go_RL_policy', 'last.pth'),
        ):
            with self.subTest(selection=selection), \
                 patch('AI.hf_hub_download', return_value=str(self.checkpoint)) as download, \
                 patch('AI.get_device', return_value=torch.device('cpu')):
                ai = ai_module.AI() if selection is None else ai_module.AI(policy_model=selection)
                download.assert_called_once_with(repo_id=repo, filename=filename, repo_type='model')
                self.assertEqual((ai.model.size, ai.model.num_blocks), (2, 1))
                self.assertFalse(ai.model.training)
                output = ai._policy_output(Game())
                self.assertEqual(tuple(output.shape), (363,))
                self.assertTrue(torch.isfinite(output).all())

    def test_invalid_ai_selection_never_downloads(self):
        for value in ('other', '', None, [], {}):
            with self.subTest(value=value), patch('AI.hf_hub_download') as download:
                with self.assertRaises(ValueError):
                    ai_module.AI(policy_model=value)
                download.assert_not_called()

    def test_resignation_logit_remains_available_for_both_models(self):
        for model in ('supervised', 'rl'):
            ai = ai_module.AI.__new__(ai_module.AI)
            ai.policy_model = model
            ai.game = Game()
            output = torch.zeros(363)
            output[362] = 100
            with self.subTest(model=model), patch.object(ai, '_policy_output', return_value=output):
                self.assertEqual(ai.predict_action(ai.game, 1), {'kind': 'resign'})
                self.assertEqual(ai.predict_action(ai.game, -1), {'kind': 'resign'})

    def test_independent_model_construction_and_reverse_choices(self):
        for black, white in (('supervised', 'rl'), ('rl', 'supervised')):
            made = []
            def make(policy_model='supervised'):
                ai = FakeAI(policy_model)
                made.append(ai)
                return ai
            with self.subTest(black=black, white=white), patch('AI.AI', side_effect=make) as constructor:
                state = self.session.initialize(0, 0, 6.5,
                    policy_model_black=black, policy_model_white=white)
                self.assertEqual([call.kwargs['policy_model'] for call in constructor.call_args_list], [black, white])
                self.assertIs(self.session.AI_black, made[0])
                self.assertIs(self.session.AI_white, made[1])
                self.assertEqual(state['policy_model_black'], black)
                self.assertEqual(state['policy_model_white'], white)
                self.assertIsNone(state['policy_model'])
                self.assertTrue(state['policy_model_label_black'])
                self.assertTrue(state['policy_model_label_white'])

    def test_legacy_shared_choice_and_per_side_override(self):
        state = self.session.initialize(policy_model='rl')
        self.assertEqual((state['policy_model_black'], state['policy_model_white']), ('rl', 'rl'))
        self.assertEqual(state['policy_model'], 'rl')
        state = self.session.initialize(policy_model='rl', policy_model_white='supervised')
        self.assertEqual((state['policy_model_black'], state['policy_model_white']), ('rl', 'supervised'))
        state = self.session.initialize()
        self.assertEqual((state['policy_model_black'], state['policy_model_white']), ('rl', 'supervised'))

    def test_each_invalid_side_preserves_entire_session_before_loading(self):
        self.session.play(2, 2, 1)
        before = copy.deepcopy(self.session.get_state())
        old_game = self.session.game
        for key in ('policy_model_black', 'policy_model_white'):
            for invalid in ('other', '', [], {}):
                with self.subTest(key=key, invalid=invalid), patch('AI.AI') as constructor:
                    with self.assertRaises(ValueError):
                        self.session.initialize(0, 0, 6.5, **{key: invalid})
                    constructor.assert_not_called()
                    self.assertIs(self.session.game, old_game)
                    self.assertEqual(self.session.get_state(), before)

    def test_failed_second_ai_load_preserves_both_models_and_choices(self):
        old_black = FakeAI('supervised')
        old_white = FakeAI('rl')
        session = GameSession(ai_factory=Mock(side_effect=[old_black, old_white]))
        session.initialize(0, 0, 6.5, policy_model_black='supervised', policy_model_white='rl')
        session.play(2, 2, 1)
        before = copy.deepcopy(session.get_state())
        board = session.game
        session._ai_factory = Mock(side_effect=[FakeAI('rl'), OSError('download unavailable')])
        with self.assertRaises(OSError):
            session.initialize(0, 0, 7.5, policy_model_black='rl', policy_model_white='supervised')
        self.assertIs(session.game, board)
        self.assertIs(session.AI_black, old_black)
        self.assertIs(session.AI_white, old_white)
        self.assertEqual(session.get_state(), before)

    def test_legacy_zero_argument_factory(self):
        factory = Mock(side_effect=[FakeAI('supervised'), FakeAI('rl')])
        session = GameSession(ai_factory=factory)
        state = session.initialize(0, 0, 6.5, policy_model_black='supervised', policy_model_white='rl')
        self.assertEqual(factory.call_count, 2)
        self.assertTrue(all(not call.args and not call.kwargs for call in factory.call_args_list))
        self.assertEqual(state['policy_model_white'], 'rl')
        self.assertEqual(session.game.komi, 6.5)

    def test_init_route_assigns_separate_models_and_ai_first_move(self):
        black = FakeAI('supervised', {'kind': 'move', 'row': 3, 'col': 4})
        white = FakeAI('rl')
        with patch('AI.AI', side_effect=[black, white]) as constructor:
            response = self.client.post('/session/init', json={
                'player_black': 0, 'player_white': 0, 'komi': 6.5,
                'policy_model_black': 'supervised', 'policy_model_white': 'rl',
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual([call.kwargs['policy_model'] for call in constructor.call_args_list], ['supervised', 'rl'])
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['board'][3][4], 1)
        self.assertEqual(data['next_player'], -1)
        self.assertEqual((data['policy_model_black'], data['policy_model_white']), ('supervised', 'rl'))
        self.assertEqual(self.client.get('/state').get_json()['policy_model_white'], 'rl')
        black.predict_action.assert_called_once()
        white.predict_action.assert_not_called()

    def test_black_ai_resignation_ends_game_without_another_ai_action(self):
        black = FakeAI('rl', {'kind': 'resign'})
        white = FakeAI('supervised')
        with patch('AI.AI', side_effect=[black, white]):
            response = self.client.post('/session/init', json={
                'player_black': 0, 'player_white': 0,
                'policy_model_black': 'rl', 'policy_model_white': 'supervised',
            })
        data = response.get_json()
        self.assertTrue(data['game_over'])
        self.assertEqual(data['result'], 'W+R')
        self.assertTrue(data['ai_move']['resign'])
        self.assertEqual(data['history'][-1]['color'], 1)
        self.assertEqual(self.client.post('/ai_move').status_code, 400)
        black.predict_action.assert_called_once()
        white.predict_action.assert_not_called()

    def test_white_ai_resignation_ends_self_play_immediately(self):
        black = FakeAI('supervised', {'kind': 'pass'})
        white = FakeAI('rl', {'kind': 'resign'})
        with patch('AI.AI', side_effect=[black, white]):
            response = self.client.post('/session/init', json={
                'player_black': 0, 'player_white': 0,
                'policy_model_black': 'supervised', 'policy_model_white': 'rl',
            })
        self.assertFalse(response.get_json()['game_over'])
        response = self.client.post('/ai_move')
        data = response.get_json()
        self.assertTrue(data['game_over'])
        self.assertEqual(data['result'], 'B+R')
        self.assertTrue(data['move']['resign'])
        self.assertEqual(data['history'][-1]['color'], -1)
        self.assertEqual(self.client.post('/ai_move').status_code, 400)
        black.predict_action.assert_called_once()
        white.predict_action.assert_called_once()

    def test_human_recommendations_follow_each_side_and_cache_by_model(self):
        made = []
        def make(policy_model='supervised'):
            ai = FakeAI(policy_model)
            made.append(ai)
            return ai
        with patch('AI.AI', side_effect=make) as constructor:
            response = self.client.post('/session/init', json={
                'policy_model_black': 'supervised', 'policy_model_white': 'rl',
            })
            self.assertEqual(response.status_code, 200)
            constructor.assert_not_called()
            black = self.client.get('/ai/recommendations').get_json()
            self.assertEqual(black['recommendations'][0]['row'], 3)
            self.session.pass_turn()
            white = self.client.get('/ai/recommendations').get_json()
            self.assertEqual(white['recommendations'][0]['row'], 7)
            self.assertEqual([ai.policy_model for ai in made], ['supervised', 'rl'])
            self.session.play(5, 5, -1)
            self.assertEqual(self.client.get('/ai/recommendations').status_code, 200)
            self.assertEqual(constructor.call_count, 2)
            self.client.post('/session/init', json={
                'policy_model_black': 'rl', 'policy_model_white': 'supervised',
            })
            self.assertEqual(self.client.get('/ai/recommendations').status_code, 200)
            self.assertEqual(constructor.call_count, 3)
            self.assertEqual(made[-1].policy_model, 'rl')

    def test_human_recommendation_does_not_reuse_other_sides_different_model(self):
        white_ai = FakeAI('rl')
        black_recommendation = FakeAI('supervised')
        with patch('AI.AI', side_effect=[white_ai, black_recommendation]) as constructor:
            self.client.post('/session/init', json={
                'player_black': 1, 'player_white': 0,
                'policy_model_black': 'supervised', 'policy_model_white': 'rl',
            })
            response = self.client.get('/ai/recommendations')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['recommendations'][0]['row'], 3)
        self.assertEqual(constructor.call_count, 2)
        white_ai.recommend_moves.assert_not_called()
        black_recommendation.recommend_moves.assert_called_once()

    def test_bad_side_or_download_failure_preserves_web_game_and_cache(self):
        self.session.play(2, 2, 1)
        before = copy.deepcopy(self.session.get_state())
        old_game = self.session.game
        cached = {'supervised': FakeAI('supervised')}
        web.recommendation_ai = cached
        response = self.client.post('/session/init', json={'policy_model_white': 'other'})
        self.assertEqual(response.status_code, 400)
        with patch('AI.AI', side_effect=OSError('model cannot be downloaded')):
            response = self.client.post('/session/init', json={
                'player_black': 0, 'player_white': 1,
                'policy_model_black': 'rl', 'policy_model_white': 'supervised',
            })
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()['success'])
        self.assertIs(self.session.game, old_game)
        self.assertIs(web.recommendation_ai, cached)
        self.assertEqual(self.client.get('/state').get_json(), before)

    def test_template_has_two_model_dropdowns_and_no_shared_selector(self):
        response = self.client.get('/')
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="policy-model-black"', html)
        self.assertIn('id="policy-model-white"', html)
        self.assertNotIn('id="policy-model"', html)
        self.assertEqual(html.count('value="rl"'), 2)
        self.assertEqual(html.count('value="supervised"'), 2)
        self.assertIn('id="session-init-message"', html)


if __name__ == '__main__':
    unittest.main()
