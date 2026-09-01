"""Binoculars scoring engine, refactored around ``LanguageProfile``.

The engine is strictly identical across languages; only the profile varies.
Documented divergences from ahans30/Binoculars (PRD §6.5):

(a) model pair and thresholds live in a declarative ``LanguageProfile``;
(b) optional shared tokenizer via ``profile.share_tokenizer_from_observer``;
(c) localised verdict labels from the profile.

``metrics.py`` is the upstream implementation, kept verbatim.
"""

from __future__ import annotations

import os
from typing import Literal, TypedDict

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

from binoculars_eu.metrics import entropy, perplexity
from binoculars_eu.profiles import DEFAULT_PROFILE_CODE, get_profile
from binoculars_eu.profiles.base import LanguageProfile
from binoculars_eu.utils import load_profile_tokenizer

type Verdict = Literal["ai", "human"]
type Confidence = Literal["low", "medium", "high"]

# Selected using Falcon-7B and Falcon-7B-Instruct at bfloat16 (Hans et al., ICML 2024).
LEGACY_ACCURACY_THRESHOLD: float = 0.9015310749276843  # optimized for f1-score
LEGACY_FPR_THRESHOLD: float = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

VALID_MODES: tuple[str, ...] = ("low-fpr", "accuracy", "tpr-at-fpr-1")

huggingface_config: dict[str, str | None] = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None),
}


class AnalyzeResult(TypedDict):
    """Structured output of ``Binoculars.analyze`` (PRD §13.1.5)."""

    score: float
    verdict: Verdict
    confidence: Confidence
    label: str
    threshold_used: float
    mode: str
    profile: str
    input_tokens: int


def _resolve_devices() -> tuple[str, str]:
    """Resolve the observer and performer devices.

    The ``DEVICE_1`` / ``DEVICE_2`` environment variables override the
    defaults; ``"auto"`` (or an absent variable) resolves to ``cuda:0`` and,
    when a second GPU exists, ``cuda:1`` — the upstream behaviour.
    """
    device_1 = os.environ.get("DEVICE_1", "auto")
    device_2 = os.environ.get("DEVICE_2", "auto")
    if device_1 == "auto":
        device_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_2 == "auto":
        device_2 = "cuda:1" if torch.cuda.device_count() > 1 else device_1
    return device_1, device_2


def _confidence(score: float, threshold: float) -> Confidence:
    """Heuristic confidence band from the relative distance to the threshold.

    The PRD requires a ``low``/``medium``/``high`` band without specifying the
    mapping; scores within ±2 % of the threshold are ``low``, within ±5 %
    ``medium``, and beyond that ``high``.
    """
    distance = abs(score - threshold) / threshold
    if distance < 0.02:
        return "low"
    if distance < 0.05:
        return "medium"
    return "high"


def _legacy_profile(observer_name_or_path: str, performer_name_or_path: str) -> LanguageProfile:
    """Build the profile reproducing upstream ahans30/Binoculars exactly.

    Falcon thresholds, English labels, strict tokenizer assertion
    (``share_tokenizer_from_observer=False``) and ``trust_remote_code=True``
    as hardcoded upstream.
    """
    return LanguageProfile(
        code="legacy",
        display_name="Upstream (Hans et al., ICML 2024)",
        observer_model=observer_name_or_path,
        performer_model=performer_name_or_path,
        threshold_accuracy=LEGACY_ACCURACY_THRESHOLD,
        threshold_low_fpr=LEGACY_FPR_THRESHOLD,
        threshold_tpr_at_fpr_1=LEGACY_FPR_THRESHOLD,
        corpus_sha256="",
        corpus_url="https://arxiv.org/abs/2401.12070",
        calibration_date="2024-01-22",
        calibration_seed=-1,
        share_tokenizer_from_observer=False,
        trust_remote_code=True,
        label_ai="Most likely AI-generated",
        label_human="Most likely human-generated",
        calibration_note=(
            "Thresholds not re-calibrated: published values from Hans et al. "
            "for Falcon-7B / Falcon-7B-Instruct in bfloat16."
        ),
    )


class Binoculars:
    """Zero-shot AI-text detector built on a language profile.

    Args:
        profile: The language profile to use; ``None`` resolves to the
            platform default profile (``fr-8b``).
        mode: Decision mode — ``"low-fpr"``, ``"accuracy"`` or
            ``"tpr-at-fpr-1"``. The threshold comes from the profile.
        max_token_observed: Maximum number of tokens scored per text.
        use_bfloat16: Load both models in bfloat16 (``float32`` otherwise).
        load_in_8bit: Load both models with bitsandbytes 8-bit quantization
            instead (PRD §16.1) — fits large pairs on 24 GB cards; mutually
            exclusive with the ``dtype`` choice above. ``None`` (default)
            defers to the profile's ``default_load_in_4bit``.
        load_in_4bit: Load both models with bitsandbytes 4-bit (nf4) — PRD
            §16.2 fallback when int8 plus activation/logit memory still
            exceeds VRAM (hybrid 8B pairs, 128k vocab); signal impact is
            measured by the calibration, never assumed. ``None`` (default)
            defers to the profile's ``default_load_in_4bit``.
    """

    def __init__(
        self,
        profile: LanguageProfile | None = None,
        mode: str = "low-fpr",
        max_token_observed: int = 512,
        use_bfloat16: bool = True,
        load_in_8bit: bool | None = None,
        load_in_4bit: bool | None = None,
    ) -> None:
        self.profile = profile if profile is not None else get_profile(DEFAULT_PROFILE_CODE)
        self.max_token_observed = max_token_observed
        self.use_bfloat16 = use_bfloat16
        self.device_1, self.device_2 = _resolve_devices()
        self.tokenizer = load_profile_tokenizer(self.profile)

        if load_in_8bit is None and load_in_4bit is None:
            # No explicit quantization: the profile decides (PRD §16.2: nf4
            # pairs like Luciole-8B cannot load in bfloat16 on target cards).
            load_in_8bit = False
            load_in_4bit = self.profile.default_load_in_4bit
        load_in_8bit = bool(load_in_8bit)
        load_in_4bit = bool(load_in_4bit)

        torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        # 8-bit (bitsandbytes) loading fits pairs like Falcon-7B on 24 GB
        # cards where two bfloat16 copies do not (PRD §16.1); dtype is then
        # managed by the quantizer, not passed to from_pretrained.
        loader_kwargs: dict = {"dtype": torch_dtype}
        if load_in_8bit:
            loader_kwargs = {"load_in_8bit": True}
        if load_in_4bit:
            loader_kwargs = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16}
        self.observer_model = AutoModelForCausalLM.from_pretrained(
            self.profile.observer_model,
            device_map={"": self.device_1},
            trust_remote_code=self.profile.trust_remote_code,
            token=huggingface_config["TOKEN"],
            **loader_kwargs,
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            self.profile.performer_model,
            device_map={"": self.device_2},
            trust_remote_code=self.profile.trust_remote_code,
            token=huggingface_config["TOKEN"],
            **loader_kwargs,
        )
        self.observer_model.eval()
        self.performer_model.eval()

        self.change_mode(mode)

    @classmethod
    def for_language(
        cls,
        code: str,
        mode: str = "low-fpr",
        max_token_observed: int = 512,
        use_bfloat16: bool = True,
        load_in_8bit: bool | None = None,
        load_in_4bit: bool | None = None,
    ) -> Binoculars:
        """Instantiate a registered profile by ISO language code.

        Raises:
            KeyError: If the profile code is not registered.
        """
        return cls(
            profile=get_profile(code),
            mode=mode,
            max_token_observed=max_token_observed,
            use_bfloat16=use_bfloat16,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )

    @classmethod
    def from_legacy(
        cls,
        observer_name_or_path: str = "tiiuae/falcon-7b",
        performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
        mode: str = "low-fpr",
        max_token_observed: int = 512,
        use_bfloat16: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> Binoculars:
        """Reproduce the exact signature and behaviour of upstream Binoculars.

        Falcon thresholds included; no registered profile required.
        """
        return cls(
            profile=_legacy_profile(observer_name_or_path, performer_name_or_path),
            mode=mode,
            max_token_observed=max_token_observed,
            use_bfloat16=use_bfloat16,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )

    def change_mode(self, mode: str) -> None:
        """Switch the decision threshold among the profile's calibrated modes.

        Raises:
            ValueError: If the mode is not one of ``VALID_MODES``.
        """
        if mode == "low-fpr":
            self.threshold = self.profile.threshold_low_fpr
        elif mode == "accuracy":
            self.threshold = self.profile.threshold_accuracy
        elif mode == "tpr-at-fpr-1":
            self.threshold = self.profile.threshold_tpr_at_fpr_1
        else:
            raise ValueError(f"Invalid mode: {mode}. Valid modes: {', '.join(VALID_MODES)}")
        self.mode = mode

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        return self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(self.observer_model.device)

    @torch.inference_mode()
    def _get_logits(
        self, encodings: transformers.BatchEncoding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observer_logits = self.observer_model(**encodings.to(self.device_1)).logits
        performer_logits = self.performer_model(**encodings.to(self.device_2)).logits
        if self.device_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    @torch.inference_mode()
    def compute_score(self, input_text: str | list[str]) -> float | list[float]:
        """Compute the Binoculars score (PPL / X-PPL) of one text or a batch."""
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits.to(self.device_1),
            performer_logits.to(self.device_1),
            encodings.to(self.device_1),
            self.tokenizer.pad_token_id,
        )
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def predict(self, input_text: str | list[str]) -> str | list[str]:
        """Return the profile-localised verdict label for one text or a batch."""
        if isinstance(input_text, str):
            score = float(self.compute_score(input_text))
            return self.profile.label_ai if score < self.threshold else self.profile.label_human
        binoculars_scores = np.array(self.compute_score(input_text))
        return np.where(
            binoculars_scores < self.threshold,
            self.profile.label_ai,
            self.profile.label_human,
        ).tolist()

    def analyze(self, input_text: str) -> AnalyzeResult:
        """Score a single text and return a structured, traceable verdict."""
        score = float(self.compute_score(input_text))
        is_ai = score < self.threshold
        return AnalyzeResult(
            score=score,
            verdict="ai" if is_ai else "human",
            confidence=_confidence(score, self.threshold),
            label=self.profile.label_ai if is_ai else self.profile.label_human,
            threshold_used=self.threshold,
            mode=self.mode,
            profile=self.profile.code,
            input_tokens=len(self.tokenizer.encode(input_text, add_special_tokens=False)),
        )
