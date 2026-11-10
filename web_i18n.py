"""Shared UI translations; model identifiers and game rules stay language-neutral."""
import json
from pathlib import Path

TRANSLATIONS = json.loads((Path(__file__).parent / 'static' / 'translations.json').read_text(encoding='utf-8'))


def normalize_language(value):
    return 'en' if value == 'en' else 'zh-CN'


def translate(message, language):
    return TRANSLATIONS.get(message, message) if language == 'en' else message
