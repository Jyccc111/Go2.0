"""Deployment/session tests never download models or contact Render."""
import io
import importlib.util
import unittest
from unittest.mock import patch

from web_sessions import BrowserSessionRegistry, SessionCapacityError

WEB = all(importlib.util.find_spec(name) for name in ('flask', 'sgfmill'))
if WEB:
    import app as web


class RegistryTests(unittest.TestCase):
    def test_active_games_are_never_evicted_and_capacity_is_bounded(self):
        registry = BrowserSessionRegistry(max_sessions=1)
        key, state, new = registry.acquire(None)
        self.assertTrue(new)
        with self.assertRaises(SessionCapacityError):
            registry.acquire(None)
        same, state2, new = registry.acquire(key)
        self.assertEqual(key, same)
        self.assertIs(state, state2)
        self.assertFalse(new)
        registry.release(key); registry.release(key)
        other, _, _ = registry.acquire(None)
        self.assertNotEqual(key, other)
        self.assertEqual(len(registry.states), 1)

    def test_idle_expiry_releases_game_but_not_active_request(self):
        now = [0.0]
        registry = BrowserSessionRegistry(max_sessions=2, ttl=10, clock=lambda:now[0])
        active, _, _ = registry.acquire(None)
        idle, _, _ = registry.acquire(None)
        registry.release(idle)
        now[0] = 11
        registry.acquire(active)
        self.assertIn(active, registry.states)
        self.assertNotIn(idle, registry.states)


@unittest.skipUnless(WEB, 'Web tests require flask and sgfmill')
class RenderSessionTests(unittest.TestCase):
    def setUp(self):
        self.config = patch.dict(web.app.config, ISOLATE_BROWSER_SESSIONS=True,
                                 SECRET_KEY='test-key-not-for-production', SESSION_COOKIE_SECURE=False)
        self.config.start(); self.addCleanup(self.config.stop)
        self.registry = BrowserSessionRegistry(max_sessions=8)
        context = patch.object(web, 'browser_sessions', self.registry)
        context.start(); self.addCleanup(context.stop)
        self.a, self.b = web.app.test_client(), web.app.test_client()

    def test_health_does_not_create_game_or_cookie(self):
        response = self.a.get('/healthz')
        self.assertEqual(response.get_json(), {'status':'ok'})
        self.assertNotIn('Set-Cookie', response.headers)
        self.assertEqual(len(self.registry.states), 0)

    def test_two_browsers_have_independent_boards_and_model_choices(self):
        original_b = self.b.get('/state').get_json()
        self.a.post('/session/init', json={'player_black':1,'player_white':1,'komi':7.5,
                                         'policy_model_black':'rl','policy_model_white':'supervised'})
        response = self.a.post('/play', json={'row':3,'col':3,'color':1})
        self.assertTrue(response.get_json()['success'])
        state_a = self.a.get('/state').get_json()
        self.assertEqual(state_a['board'][3][3], 1)
        self.assertEqual(state_a['policy_model_black'], 'rl')
        self.assertEqual(self.b.get('/state').get_json(), original_b)
        self.assertEqual(len(self.registry.states), 2)
        self.assertTrue(all(state.active == 0 for state in self.registry.states.values()))

    def test_sgf_replay_is_private_to_browser_and_language_keeps_game(self):
        self.a.post('/sgf/load', data={'sgf':(io.BytesIO(b'(;GM[1]FF[4]SZ[19]KM[6.5];B[dd])'), 'one.sgf')})
        self.a.post('/sgf/step')
        self.a.get('/?lang=en')
        self.assertEqual(self.a.get('/sgf/state').get_json()['index'], 1)
        self.assertEqual(self.b.get('/sgf/state').status_code, 400)

    def test_expired_cookie_cannot_silently_play_on_new_board(self):
        self.a.get('/state')
        self.registry.states.clear()
        response = self.a.post('/play', json={'row':3,'col':3,'color':1})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.a.get('/state').get_json()['board'][3][3], 0)

    def test_upload_limit_returns_json(self):
        response = self.a.post('/sgf/load', data=b'x'*(2*1024*1024+1), content_type='application/octet-stream')
        self.assertEqual(response.status_code, 413)
        self.assertFalse(response.get_json()['ok'])
        self.assertTrue(all(state.active == 0 for state in self.registry.states.values()))


if __name__ == '__main__':
    unittest.main()
