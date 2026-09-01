"""Tests for the ``LanguageProfile`` dataclass and the profile registry."""

import sys

import pytest

import binoculars_eu.profiles as profiles_module
from binoculars_eu.profiles import (
    DEFAULT_PROFILE_CODE,
    get_profile,
    list_profiles,
    register,
)
from binoculars_eu.profiles.base import LanguageProfile


def make_profile(**overrides: object) -> LanguageProfile:
    """Build a valid profile with sensible defaults, overridable per test."""
    params: dict[str, object] = {
        "code": "zz",
        "display_name": "Test",
        "observer_model": "org/obs",
        "performer_model": "org/perf",
        "threshold_accuracy": 0.9,
        "threshold_low_fpr": 0.85,
        "threshold_tpr_at_fpr_1": 0.85,
        "corpus_sha256": "abc",
        "corpus_url": "https://example.org/corpus",
        "calibration_date": "2026-08-30",
        "calibration_seed": 42,
    }
    params.update(overrides)
    return LanguageProfile(**params)  # type: ignore[arg-type]


class TestLanguageProfile:
    def test_defaults(self) -> None:
        p = make_profile()
        assert p.share_tokenizer_from_observer is True
        assert p.trust_remote_code is False
        assert p.label_ai == "Probablement généré par IA"
        assert p.label_human == "Probablement écrit par un humain"
        assert p.calibration_note is None

    def test_frozen_instance(self) -> None:
        p = make_profile()
        with pytest.raises(AttributeError, match="FrozenInstanceError|cannot assign"):
            p.threshold_accuracy = 0.1  # type: ignore[misc]
        assert p.threshold_accuracy == 0.9


class TestRegistry:
    def test_default_profile_code(self) -> None:
        assert DEFAULT_PROFILE_CODE == "fr-8b"

    def test_auto_discovery_finds_all_profiles(self) -> None:
        assert sorted(p.code for p in list_profiles()) == ["fr", "fr-8b"]

    def test_fr8b_profile_contract(self) -> None:
        from binoculars_eu.profiles import profile_dir

        p = get_profile("fr-8b")
        assert p.observer_model == "OpenLLM-France/Luciole-8B-Base"
        assert p.performer_model == "OpenLLM-France/Luciole-8B-Instruct-1.1"
        assert p.trust_remote_code is True
        assert p.default_load_in_4bit is True
        # code "fr-8b" lives in the fr8b package dir (no "-" in module names)
        assert profile_dir("fr-8b").name == "fr8b"

    def test_get_profile_fr(self) -> None:
        p = get_profile("fr")
        assert p.code == "fr"
        assert p.observer_model == "OpenLLM-France/Luciole-1B-Base"
        assert p.performer_model == "OpenLLM-France/Luciole-1B-SFT-1.0"

    def test_unknown_profile_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="Unknown profile"):
            get_profile("de")

    def test_register_and_duplicate_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry: dict[str, LanguageProfile] = {}
        monkeypatch.setattr(profiles_module, "_REGISTRY", registry)
        fresh = make_profile(code="zz")
        assert register(fresh) is fresh
        assert registry["zz"] is fresh
        with pytest.raises(ValueError, match="already registered"):
            register(fresh)

    def test_discover_skips_base_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry: dict[str, LanguageProfile] = {}
        monkeypatch.setattr(profiles_module, "_REGISTRY", registry)
        # The fr subpackage registers at IMPORT time; purge it so _discover()
        # re-imports it and the register() side effect fires again.
        monkeypatch.delitem(
            sys.modules, "binoculars_eu.profiles.fr", raising=False
        )
        profiles_module._discover()
        assert "fr" in registry
        assert "base" not in registry  # base.py is a module, not a profile package
