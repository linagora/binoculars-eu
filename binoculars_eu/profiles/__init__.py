"""Language profile registry with auto-discovery.

Each profile is a package under ``binoculars_eu/profiles/`` that registers
itself at import time. The registry discovers profiles with
``pkgutil.iter_modules``: adding a language means adding a directory.
"""

from __future__ import annotations

import importlib
import pkgutil

from binoculars_eu.profiles.base import LanguageProfile

_REGISTRY: dict[str, LanguageProfile] = {}
DEFAULT_PROFILE_CODE: str = "fr"


def register(profile: LanguageProfile) -> LanguageProfile:
    """Register a language profile; called by each ``profiles/<lang>/__init__.py``.

    Args:
        profile: The language profile to register.

    Returns:
        The registered profile (chainable).

    Raises:
        ValueError: If a profile with the same code is already registered.
    """
    if profile.code in _REGISTRY:
        raise ValueError(f"Profile already registered: {profile.code}")
    _REGISTRY[profile.code] = profile
    return profile


def _discover() -> None:
    """Import every subpackage of ``profiles/`` to trigger its ``register()``."""
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.ispkg:  # base.py is not a package
            importlib.import_module(f"{__name__}.{module_info.name}")


def get_profile(code: str) -> LanguageProfile:
    """Return a registered profile by ISO language code.

    Args:
        code: ISO 639-1 language code, e.g. ``"fr"``.

    Returns:
        The requested language profile.

    Raises:
        KeyError: If the profile code is unknown; the error message lists the
            available profiles.
    """
    if not _REGISTRY:
        _discover()
    if code not in _REGISTRY:
        available = sorted(_REGISTRY)
        raise KeyError(f"Unknown profile: {code!r}. Available: {available}")
    return _REGISTRY[code]


def list_profiles() -> list[LanguageProfile]:
    """Return all registered profiles, sorted by language code."""
    if not _REGISTRY:
        _discover()
    return [_REGISTRY[c] for c in sorted(_REGISTRY)]
