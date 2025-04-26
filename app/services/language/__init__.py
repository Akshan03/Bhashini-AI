# This file makes the language directory a Python package
from .detection import language_detector
from .translation import translator
from .supported import (
    SUPPORTED_LANGUAGES,
    LANGUAGE_CODES,
    get_language_details,
    is_language_supported,
    get_language_name
)

__all__ = [
    'language_detector',
    'translator',
    'SUPPORTED_LANGUAGES',
    'LANGUAGE_CODES',
    'get_language_details',
    'is_language_supported',
    'get_language_name'
]
