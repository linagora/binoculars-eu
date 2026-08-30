"""binoculars-eu: European multilingual zero-shot AI-text detection platform.

The public API exposes ``LanguageProfile`` and the profile registry. The
``Binoculars`` detector class is added in phase 2.
"""

from binoculars_eu.profiles import get_profile, list_profiles
from binoculars_eu.profiles.base import LanguageProfile

__version__ = "0.1.0"

__all__ = [
    "LanguageProfile",
    "__version__",
    "get_profile",
    "list_profiles",
]
