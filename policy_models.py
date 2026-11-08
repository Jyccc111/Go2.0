"""Web policy choices, shared without importing a model or starting training."""
POLICY_MODELS = {
    'supervised': {
        'label': '监督学习模型',
        'repo_id': 'Jycccc111/Go-Policy',
        'filename': 'best_policy_network_128.pth',
    },
    'rl': {
        'label': '强化学习模型（最新）',
        'repo_id': 'Jycccc111/Go_RL_policy',
        'filename': 'last.pth',
    },
}


def validate_policy_model(value):
    if not isinstance(value, str) or value not in POLICY_MODELS:
        raise ValueError('请选择监督学习模型或强化学习模型。')
    return value
