"""long-output-v1: six attack-free long-form specs that expand/cycle
correctly and format against dataset records with the length instruction."""

from __future__ import annotations

import pytest

from app.core import (
    PROMPTBENCH_PRESETS,
    _expand_promptbench_preset,
)

LENGTH_SENTENCE = "Write a thorough, detailed response of at least 1200 words."


def test_preset_shape():
    preset = PROMPTBENCH_PRESETS["long-output-v1"]
    assert len(preset) == 6
    assert [s.dataset for s in preset] == [
        "gsm8k", "gsm8k", "sst2", "sst2", "cola", "cola"
    ]
    assert all(s.attack is None for s in preset)
    assert all(s.template.rstrip().endswith(LENGTH_SENTENCE) for s in preset)
    prefixes = [s.prompt_id for s in preset]
    assert prefixes == [
        "pb-gsm8k-long-0000-tutorial",
        "pb-gsm8k-long-0001-teaching",
        "pb-sst2-long-0000-essay",
        "pb-sst2-long-0001-rewrite",
        "pb-cola-long-0000-grammar",
        "pb-cola-long-0001-lesson",
    ]


def test_expansion_cycles_dataset_indices():
    expanded = _expand_promptbench_preset("long-output-v1", 13)
    assert len(expanded) == 13
    # Unique ids across cycles.
    assert len({s.prompt_id for s in expanded}) == 13
    # Second cycle advances each spec's dataset index by 1.
    assert expanded[6].dataset_index == expanded[0].dataset_index + 1
    assert expanded[6].prompt_id == "pb-gsm8k-long-0000-tutorial-0001"


def test_templates_format_with_content_field():
    # Every template must consume the {content} field (the PromptBench
    # record key all three datasets provide) and keep the length sentence.
    for spec in PROMPTBENCH_PRESETS["long-output-v1"]:
        rendered = spec.template.format(content="SAMPLE-CONTENT")
        assert "SAMPLE-CONTENT" in rendered
        assert rendered.rstrip().endswith(LENGTH_SENTENCE)


def test_verification_v1_untouched():
    assert len(PROMPTBENCH_PRESETS["verification-v1"]) == 6
