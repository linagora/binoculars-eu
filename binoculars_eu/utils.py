"""Tokenizer utilities.

Keeps the strict upstream consistency assertion unchanged and provides the
profile-aware loader implementing the ``share_tokenizer_from_observer``
relaxation (PRD §6.5, divergence b).
"""

from __future__ import annotations

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from binoculars_eu.profiles.base import LanguageProfile


def assert_tokenizer_consistency(model_id_1: str, model_id_2: str) -> None:
    """Strict upstream check: both models must have identical vocabularies.

    Args:
        model_id_1: HuggingFace repo of the first model.
        model_id_2: HuggingFace repo of the second model.

    Raises:
        ValueError: If the two tokenizers' vocabularies differ.
    """
    identical_tokenizers = (
        AutoTokenizer.from_pretrained(model_id_1).vocab
        == AutoTokenizer.from_pretrained(model_id_2).vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")


def load_profile_tokenizer(profile: LanguageProfile) -> PreTrainedTokenizerBase:
    """Load the tokenizer for a profile according to ``share_tokenizer_from_observer``.

    When ``True``, a single tokenizer is loaded from the observer model and
    shared by both models: the strict upstream assertion becomes unnecessary
    (the tokenizers are functionally interchangeable) and ~500 MB of RAM are
    saved. When ``False``, the strict upstream assertion is applied unchanged
    before loading — this is the path used by ``Binoculars.from_legacy``.

    Args:
        profile: The language profile being instantiated.

    Returns:
        The loaded tokenizer, with ``pad_token`` set to ``eos_token`` if unset.
    """
    if profile.share_tokenizer_from_observer:
        tokenizer = AutoTokenizer.from_pretrained(profile.observer_model)
    else:
        assert_tokenizer_consistency(profile.observer_model, profile.performer_model)
        tokenizer = AutoTokenizer.from_pretrained(profile.observer_model)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
