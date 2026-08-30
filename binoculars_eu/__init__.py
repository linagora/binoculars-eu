"""binoculars-eu: European multilingual zero-shot AI-text detection platform.

The public API exposes the ``Binoculars`` detector, the ``LanguageProfile``
dataclass and the profile registry.
"""

from binoculars_eu.detector import AnalyzeResult, Binoculars
from binoculars_eu.profiles import get_profile, list_profiles
from binoculars_eu.profiles.base import LanguageProfile

__version__ = "0.1.0"

__all__ = [
    "AnalyzeResult",
    "Binoculars",
    "LanguageProfile",
    "__version__",
    "get_profile",
    "list_profiles",
]
