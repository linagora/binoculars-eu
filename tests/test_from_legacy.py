"""Upstream-compatibility tests: ``from_legacy`` parity, metrics verbatim,
tokenizer utilities. No model weights are ever loaded — HF loaders are
monkeypatched with stubs.

Structural invariant under test (PRD §12, invariant 1): ``metrics.py`` stays
identical to ahans30/Binoculars so upstream rebases remain conflict-free.
"""

import ast
import hashlib
import inspect
from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding

import binoculars_eu.detector as detector_module
import binoculars_eu.utils as utils_module
from binoculars_eu.detector import (
    LEGACY_ACCURACY_THRESHOLD,
    LEGACY_FPR_THRESHOLD,
    Binoculars,
    _confidence,
    _legacy_profile,
)

# SHA-256 of the AST of perplexity()+entropy() as copied from upstream
# (ahans30/Binoculars @ main, fetched 2026-08-30). Refresh only after a
# verified upstream sync — a mismatch means the math diverged.
METRICS_AST_SHA256 = "8234c994f8a97b6f9f39f099a64d1da2ebab5c086928ff0270e992bdb1b7cdc0"

FALCON_ACCURACY = 0.9015310749276843
FALCON_FPR = 0.8536432310785527


# --------------------------------------------------------------------------
# Fixtures: weightless construction of a legacy detector
# --------------------------------------------------------------------------
@pytest.fixture()
def legacy_detector(monkeypatch: pytest.MonkeyPatch) -> Binoculars:
    tokenizer = SimpleNamespace(
        pad_token_id=1,
        eos_token="<eos>",
        encode=lambda text, add_special_tokens=False: text.split(),
    )
    monkeypatch.setattr(
        detector_module, "load_profile_tokenizer", lambda profile: tokenizer
    )

    class FakeModel:
        device = "cpu"

        def eval(self) -> "FakeModel":
            return self

    monkeypatch.setattr(
        detector_module.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: FakeModel()),
    )
    return Binoculars.from_legacy(mode="low-fpr")


# --------------------------------------------------------------------------
# _legacy_profile parity with the upstream paper constants
# --------------------------------------------------------------------------
class TestLegacyProfile:
    def test_thresholds_match_hans_et_al(self) -> None:
        profile = _legacy_profile("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct")
        assert profile.threshold_accuracy == LEGACY_ACCURACY_THRESHOLD == FALCON_ACCURACY
        assert profile.threshold_low_fpr == LEGACY_FPR_THRESHOLD == FALCON_FPR
        assert profile.threshold_tpr_at_fpr_1 == FALCON_FPR

    def test_model_pair_defaults(self) -> None:
        profile = _legacy_profile("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct")
        assert profile.observer_model == "tiiuae/falcon-7b"
        assert profile.performer_model == "tiiuae/falcon-7b-instruct"

    def test_upstream_flags(self) -> None:
        profile = _legacy_profile("obs", "perf")
        # Strict tokenizer assertion path, as upstream.
        assert profile.share_tokenizer_from_observer is False
        # Upstream hardcodes trust_remote_code=True.
        assert profile.trust_remote_code is True

    def test_english_labels(self) -> None:
        profile = _legacy_profile("obs", "perf")
        assert profile.label_ai == "Most likely AI-generated"
        assert profile.label_human == "Most likely human-generated"

    def test_traceability_fields(self) -> None:
        profile = _legacy_profile("obs", "perf")
        assert profile.code == "legacy"
        assert profile.calibration_seed == -1
        assert profile.calibration_date == "2024-01-22"
        assert profile.calibration_note


# --------------------------------------------------------------------------
# from_legacy wiring (no weights)
# --------------------------------------------------------------------------
class TestFromLegacy:
    def test_default_mode_uses_fpr_threshold(self, legacy_detector: Binoculars) -> None:
        assert legacy_detector.threshold == FALCON_FPR
        assert legacy_detector.mode == "low-fpr"

    def test_change_mode_accuracy(self, legacy_detector: Binoculars) -> None:
        legacy_detector.change_mode("accuracy")
        assert legacy_detector.threshold == FALCON_ACCURACY
        assert legacy_detector.mode == "accuracy"

    def test_change_mode_tpr_at_fpr_1(self, legacy_detector: Binoculars) -> None:
        legacy_detector.change_mode("tpr-at-fpr-1")
        # Upstream has no such threshold; legacy reuses the FPR value.
        assert legacy_detector.threshold == FALCON_FPR

    def test_change_mode_invalid(self, legacy_detector: Binoculars) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            legacy_detector.change_mode("bogus")

    def test_predict_english_labels(
        self, legacy_detector: Binoculars, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(legacy_detector, "compute_score", lambda text: 0.7)
        assert legacy_detector.predict("any text") == "Most likely AI-generated"
        monkeypatch.setattr(legacy_detector, "compute_score", lambda text: 0.99)
        assert legacy_detector.predict("any text") == "Most likely human-generated"

    def test_predict_batch(
        self, legacy_detector: Binoculars, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(legacy_detector, "compute_score", lambda text: [0.7, 0.99])
        assert legacy_detector.predict(["a", "b"]) == [
            "Most likely AI-generated",
            "Most likely human-generated",
        ]

    def test_analyze_structure(
        self, legacy_detector: Binoculars, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(legacy_detector, "compute_score", lambda text: 0.7)
        result = legacy_detector.analyze("un deux trois quatre")
        assert result["verdict"] == "ai"
        assert result["profile"] == "legacy"
        assert result["mode"] == "low-fpr"
        assert result["threshold_used"] == FALCON_FPR
        assert result["input_tokens"] == 4

    def test_for_language_unknown_profile(self) -> None:
        with pytest.raises(KeyError, match="Unknown profile"):
            Binoculars.for_language("de")

    def test_devices_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEVICE_1", "cpu")
        monkeypatch.setenv("DEVICE_2", "cpu")
        assert detector_module._resolve_devices() == ("cpu", "cpu")


class TestConfidence:
    def test_bands(self) -> None:
        assert _confidence(0.90, 0.90) == "low"      # < 2 % from threshold
        assert _confidence(0.90, 0.92) == "medium"   # ~2.2 %
        assert _confidence(0.70, 0.90) == "high"


# --------------------------------------------------------------------------
# metrics: functional smoke + verbatim-upstream invariant
# --------------------------------------------------------------------------
class TestMetrics:
    def test_perplexity_and_entropy_shapes(self) -> None:
        from binoculars_eu.metrics import entropy, perplexity

        generator = torch.Generator().manual_seed(0)
        enc = BatchEncoding({
            "input_ids": torch.tensor([[5, 8, 2, 9, 1, 1, 1, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]),
        })
        logits = torch.randn(1, 8, 16, generator=generator)
        ppl = perplexity(enc, logits)
        x_ppl = entropy(logits, logits, enc, pad_token_id=1)
        assert ppl.shape == (1,)
        assert x_ppl.shape == (1,)
        assert float(ppl[0]) > 0 and float(x_ppl[0]) > 0

    def test_median_variants(self) -> None:
        from binoculars_eu.metrics import entropy, perplexity

        generator = torch.Generator().manual_seed(1)
        enc = BatchEncoding({
            "input_ids": torch.tensor([[5, 8, 2, 9, 3, 4, 1, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
        })
        logits = torch.randn(1, 8, 16, generator=generator)
        assert perplexity(enc, logits, median=True).shape == (1,)
        assert entropy(logits, logits, enc, pad_token_id=1, median=True).shape == (1,)

    def test_metrics_verbatim_vs_upstream(self) -> None:
        from binoculars_eu import metrics

        src = inspect.getsource(metrics.perplexity) + inspect.getsource(metrics.entropy)
        tree = ast.parse(src)
        digest = hashlib.sha256(
            ast.dump(tree, annotate_fields=False).encode()
        ).hexdigest()
        assert digest == METRICS_AST_SHA256, (
            "metrics.py diverged from ahans30/Binoculars — restore verbatim "
            "code, or refresh METRICS_AST_SHA256 after a verified upstream sync"
        )


# --------------------------------------------------------------------------
# utils: tokenizer consistency + profile-aware loader
# --------------------------------------------------------------------------
class TestTokenizerUtils:
    def test_assert_consistency_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        vocabs = {"org/a": {"x": 0}, "org/b": {"x": 0}, "org/c": {"y": 0}}

        class FakeTokenizer:
            def __init__(self, vocab: dict) -> None:
                self.vocab = vocab

        monkeypatch.setattr(
            utils_module.AutoTokenizer,
            "from_pretrained",
            staticmethod(lambda model_id, **kwargs: FakeTokenizer(vocabs[model_id])),
        )
        utils_module.assert_tokenizer_consistency("org/a", "org/b")  # no raise
        with pytest.raises(ValueError, match="not identical"):
            utils_module.assert_tokenizer_consistency("org/a", "org/c")

    def test_load_shared_tokenizer_single_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loaded: list[str] = []

        class FakeTokenizer:
            pad_token = None
            eos_token = "<eos>"

        def fake_from_pretrained(model_id: str, **kwargs: object) -> FakeTokenizer:
            loaded.append(model_id)
            return FakeTokenizer()

        monkeypatch.setattr(
            utils_module.AutoTokenizer, "from_pretrained",
            staticmethod(fake_from_pretrained),
        )
        profile = SimpleNamespace(
            observer_model="org/obs",
            performer_model="org/perf",
            share_tokenizer_from_observer=True,
        )
        tokenizer = utils_module.load_profile_tokenizer(profile)  # type: ignore[arg-type]
        assert loaded == ["org/obs"]  # one load only: observer tokenizer shared
        assert tokenizer.pad_token == "<eos>"  # pad falls back to eos

    def test_load_strict_tokenizer_runs_assertion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loaded: list[str] = []

        class FakeTokenizer:
            pad_token = "<pad>"
            vocab = {"x": 0}  # needed by the strict consistency assertion

        def fake_from_pretrained(model_id: str, **kwargs: object) -> FakeTokenizer:
            loaded.append(model_id)
            return FakeTokenizer()

        monkeypatch.setattr(
            utils_module.AutoTokenizer, "from_pretrained",
            staticmethod(fake_from_pretrained),
        )
        profile = SimpleNamespace(
            observer_model="org/obs",
            performer_model="org/perf",
            share_tokenizer_from_observer=False,
        )
        tokenizer = utils_module.load_profile_tokenizer(profile)  # type: ignore[arg-type]
        # consistency check (obs + perf) then observer tokenizer load.
        assert loaded == ["org/obs", "org/perf", "org/obs"]
        assert tokenizer.pad_token == "<pad>"
