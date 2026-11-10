"""Bilingual views share game/model state; replay reads never advance a move."""
import io
import importlib.util
import re
import unittest
from unittest.mock import patch
from html.parser import HTMLParser

from policy_models import get_policy_model
from web_i18n import TRANSLATIONS

WEB = all(importlib.util.find_spec(name) for name in ('flask', 'sgfmill'))
if WEB:
    import app as web
    from GameSession import GameSession


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = 0
        self.text = []

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)


@unittest.skipUnless(WEB, 'Web tests require flask and sgfmill')
class BilingualWebTests(unittest.TestCase):
    def setUp(self):
        self.session = GameSession()
        self.globals = patch.multiple(web, game_session=self.session,
                                      game=self.session.game, sgf_replay=None,
                                      recommendation_ai={})
        self.globals.start()
        self.addCleanup(self.globals.stop)
        self.client = web.app.test_client()

    def test_chinese_and_english_pages_have_translated_controls_and_labels(self):
        chinese = self.client.get('/?lang=zh-CN').get_data(as_text=True)
        english = self.client.get('/?lang=en').get_data(as_text=True)
        self.assertIn('<html lang="zh-CN">', chinese)
        self.assertIn('<html lang="en">', english)
        self.assertIn('新对局', chinese)
        visible = VisibleText(); visible.feed(english)
        text = ' '.join(visible.text).replace('中文', '')
        self.assertIsNone(re.search(r'[\u3400-\u9fff]', text), text)
        self.assertIn('Black AI model', text)
        self.assertIn('White AI model', text)
        self.assertIn('SGF and win-rate analysis', text)
        self.assertIn('Latest loads Go_RL_policy/last.pth', text)
        self.assertIn('Slow AI self-play', text)
        self.assertIn('Move interval', text)
        self.assertIn('static/ai_playback.js', english)
        self.assertIn('static/i18n.js', english)

    def test_changing_language_never_reinitializes_board_or_model_choices(self):
        self.session.initialize(player_black=1, player_white=1,
                                policy_model_black='rl', policy_model_white='supervised')
        self.session.play(3, 3, 1)
        before = self.session.get_state()
        for language in ('en', 'zh-CN', 'en'):
            self.assertEqual(self.client.get('/?lang=' + language).status_code, 200)
            self.assertEqual(self.client.get('/state').get_json(), before)

    def test_language_cookie_and_invalid_values(self):
        self.client.set_cookie('go_language', 'en')
        self.assertIn('<html lang="en">', self.client.get('/').get_data(as_text=True))
        self.assertIn('<html lang="zh-CN">', self.client.get('/?lang=zh-CN').get_data(as_text=True))
        response = self.client.get('/?lang=%3Cscript%3Ealert(1)%3C/script%3E').get_data(as_text=True)
        self.assertIn('<html lang="zh-CN">', response)
        self.assertNotIn('<script>alert(1)</script>', response)

    def test_sgf_state_read_preserves_replay_position_across_languages(self):
        self.assertEqual(self.client.get('/sgf/state').status_code, 400)
        loaded = self.client.post('/sgf/load', data={
            'sgf': (io.BytesIO(b'(;GM[1]FF[4]SZ[19]KM[6.5]RE[B+R];B[dd];W[pp])'), 'sample.sgf')})
        self.assertEqual(loaded.status_code, 200)
        self.client.post('/sgf/step')
        before = self.client.get('/sgf/state').get_json()
        self.assertEqual(before['index'], 1)
        for language in ('en', 'zh-CN'):
            self.client.get('/?lang=' + language)
            self.assertEqual(self.client.get('/sgf/state').get_json(), before)

    def test_latest_option_is_root_alias_not_largest_epoch(self):
        source = get_policy_model('rl')
        self.assertEqual(source['repo_id'], 'Jycccc111/Go_RL_policy')
        self.assertEqual(source['filename'], 'last.pth')
        self.assertNotIn('revision', source)
        self.assertIn('强化学习模型（最新）', TRANSLATIONS)


if __name__ == '__main__':
    unittest.main()
