"""Epoch selection contract; all remote metadata and weight downloads are mocked."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
import policy_models as catalog
from Train_net import Policy_network

WEB = all(importlib.util.find_spec(name) for name in ('flask', 'sgfmill'))
if WEB:
    import AI as ai_module
    import app as web
    from GameSession import GameSession


def epoch_file(number, run='20261108T000000Z-batch001', digest='a' * 64):
    return NS(rfilename=f'runs/{run}/epochs/epoch-{number:06d}/policy.pth',
              lfs=NS(sha256=digest), blob_id='blob')


class CatalogTestBase(unittest.TestCase):
    def setUp(self):
        self.cache = patch.multiple(catalog, _CATALOG=None, _CATALOG_TIME=0.0)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.registry = patch.dict(catalog._EPOCH_MODELS, {}, clear=True)
        self.registry.start()
        self.addCleanup(self.registry.stop)

    def build(self, files, revision='1' * 40):
        api = Mock()
        api.model_info.return_value = NS(sha=revision, siblings=files)
        return catalog.list_policy_models(refresh=True, api=api), api


class EpochCatalogTests(CatalogTestBase):
    def test_lists_real_epoch_files_and_disambiguates_runs(self):
        rows, api = self.build([epoch_file(1), epoch_file(2),
            epoch_file(1, run='20261109T000000Z-batch002', digest='b' * 64),
            NS(rfilename='last.pth'), NS(rfilename='runs/old/policy-20.pth'),
            NS(rfilename='../epochs/epoch-1/policy.pth')])
        epochs = [row for row in rows if row.get('epoch')]
        self.assertEqual(len(epochs), 3)
        self.assertEqual([row['id'] for row in rows[:2]], ['supervised', 'rl'])
        self.assertEqual(len({row['id'] for row in epochs}), 3)
        self.assertEqual(len({row['label'] for row in epochs}), 3)
        self.assertEqual(epochs[0]['epoch'], 2)
        self.assertFalse(any('批次' in row['label'] for row in epochs))
        api.model_info.assert_called_once_with(repo_id='Jycccc111/Go_RL_policy', files_metadata=True)

    def test_refresh_keeps_selected_version_pinned_and_ids_stable(self):
        rows, _ = self.build([epoch_file(1)])
        old = rows[-1]
        refreshed, _ = self.build([epoch_file(2), epoch_file(1)], revision='2' * 40)
        same = next(row for row in refreshed if row.get('epoch') == 1)
        self.assertEqual(same['id'], old['id'])
        self.assertEqual(catalog.get_policy_model(old['id'])['revision'], '1' * 40)
        self.build([epoch_file(3)], revision='3' * 40)
        self.assertEqual(catalog.get_policy_model(old['id'])['revision'], '1' * 40)

    def test_cache_returns_copies_and_unknown_ids_are_rejected(self):
        rows, _ = self.build([epoch_file(1)])
        rows[-1]['label'] = 'changed by client'
        api = Mock()
        cached = catalog.list_policy_models(api=api)
        self.assertNotEqual(cached[-1]['label'], 'changed by client')
        api.model_info.assert_not_called()
        for invalid in ('rl-epoch-unregistered', '../../weights.pth', None, {}):
            with self.assertRaises(ValueError):
                catalog.validate_policy_model(invalid)

    def test_flat_and_legacy_copies_share_one_choice_without_dates(self):
        digest = 'a' * 64
        rows, _ = self.build([epoch_file(5),
            NS(rfilename=f'models/epoch-000005-{digest}.pth', lfs=NS(sha256=digest)),
            NS(rfilename=f'models/policy-{digest}.pth', lfs=NS(sha256=digest)),
            NS(rfilename='models/policy-' + 'b' * 64 + '.pth', lfs=NS(sha256='b' * 64))])
        versions = [row for row in rows if row.get('version')]
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[0]['epoch'], 5)
        self.assertTrue(versions[0]['filename'].startswith('models/'))
        self.assertNotIn('2026', versions[0]['label'])


@unittest.skipUnless(WEB, 'Web tests require flask and sgfmill')
class EpochWebTests(CatalogTestBase):
    def setUp(self):
        super().setUp()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.checkpoint = Path(self.directory.name) / 'model.pth'
        torch.set_num_threads(1)
        torch.save(Policy_network(size=2, num_blocks=0).state_dict(), self.checkpoint)
        digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        rows, _ = self.build([epoch_file(1, digest=digest), epoch_file(2, digest=digest)])
        self.ids = {row['epoch']: row['id'] for row in rows if row.get('epoch')}
        self.session = GameSession()
        globals_patch = patch.multiple(web, game_session=self.session, game=self.session.game,
                                       recommendation_ai={}, sgf_replay=None)
        globals_patch.start()
        self.addCleanup(globals_patch.stop)
        self.client = web.app.test_client()

    def test_loader_passes_exact_epoch_path_and_commit(self):
        for epoch, key in self.ids.items():
            source = catalog.get_policy_model(key)
            with patch('AI.hf_hub_download', return_value=str(self.checkpoint)) as download, \
                 patch('AI.get_device', return_value=torch.device('cpu')):
                ai = ai_module.AI(policy_model=key)
            download.assert_called_once_with(repo_id='Jycccc111/Go_RL_policy',
                filename=source['filename'], repo_type='model', revision='1' * 40)
            self.assertEqual(ai.policy_source['epoch'], epoch)
            self.assertFalse(ai.model.training)

    def test_black_and_white_load_independent_epoch_choices(self):
        def make(policy_model):
            ai = Mock(policy_model=policy_model)
            ai.predict_action.return_value = {'kind': 'pass'}
            return ai
        with patch('AI.AI', side_effect=make) as make_ai:
            response = self.client.post('/session/init', json={
                'player_black': 0, 'player_white': 0,
                'policy_model_black': self.ids[1], 'policy_model_white': self.ids[2]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([call.kwargs['policy_model'] for call in make_ai.call_args_list],
                         [self.ids[1], self.ids[2]])
        state = response.get_json()
        self.assertIn('Epoch 1', state['policy_model_label_black'])
        self.assertIn('Epoch 2', state['policy_model_label_white'])
        self.assertIsNone(state['policy_model'])

    def test_recommendation_cache_distinguishes_epochs(self):
        self.session.initialize(policy_model_black=self.ids[1], policy_model_white=self.ids[2])
        def make(policy_model):
            ai = Mock(policy_model=policy_model)
            ai.recommend_moves.return_value = []
            return ai
        with patch('AI.AI', side_effect=make) as create:
            self.assertEqual(self.client.get('/ai/recommendations').status_code, 200)
            self.session.pass_turn()
            self.assertEqual(self.client.get('/ai/recommendations').status_code, 200)
        self.assertEqual([call.kwargs['policy_model'] for call in create.call_args_list],
                         [self.ids[1], self.ids[2]])

    def test_catalog_failure_and_unknown_choice_preserve_game(self):
        self.session.play(3, 3, 1)
        before = self.session.get_state()
        with patch('app.list_policy_models', side_effect=ConnectionError('offline')):
            response = self.client.get('/policy/models?refresh=1')
        self.assertEqual(response.status_code, 503)
        self.assertEqual([row['id'] for row in response.get_json()['models']], ['supervised', 'rl'])
        response = self.client.post('/session/init', json={'policy_model_white': 'rl-epoch-unknown'})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.session.get_state(), before)

    def test_catalog_api_and_refresh_control(self):
        with patch('app.list_policy_models', return_value=catalog.list_policy_models()) as listing:
            response = self.client.get('/policy/models?refresh=1')
        self.assertEqual(response.status_code, 200)
        listing.assert_called_once_with(refresh=True)
        self.assertEqual(len(response.get_json()['models']), 4)
        html = self.client.get('/').get_data(as_text=True)
        self.assertIn('id="refresh-policy-models"', html)
        self.assertIn('id="policy-model-black"', html)
        self.assertIn('id="policy-model-white"', html)


if __name__ == '__main__':
    unittest.main()
