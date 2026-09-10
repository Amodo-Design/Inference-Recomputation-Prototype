"""Pure tests for content-derived model identity (app/identity.py).

No database — these guard the determinism invariant: if the namespace or the
canonical serialisation ever changes, every existing model_id silently re-keys
and dedup/resolution break. The golden-value tests are the tripwire for that.
"""

from __future__ import annotations

import pytest

from app.identity import derive_model_id

BASE = dict(
    model_name="m",
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    seed=1,
    decoding_algorithm="gumbel_max",
)


def test_deterministic_across_calls():
    assert derive_model_id(**BASE) == derive_model_id(**BASE)


def test_golden_value_minimal():
    """Name-only config pins to a fixed UUID (namespace/serialisation tripwire)."""
    got = derive_model_id(
        model_name="golden-model",
        temperature=None,
        top_k=None,
        top_p=None,
        seed=None,
        decoding_algorithm=None,
    )
    assert str(got) == "f7149504-a2ca-5b6a-8a98-ca36d047eaa1"


def test_golden_value_full_config():
    got = derive_model_id(
        model_name="openai/gpt-oss-20b",
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        seed=42,
        decoding_algorithm="gumbel_max",
    )
    assert str(got) == "3db96698-c03b-55b0-ac81-5ce6cf43a21f"


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("model_name", "m2"),
        ("temperature", 0.8),
        ("top_k", 51),
        ("top_p", 0.91),
        ("seed", 2),
        ("decoding_algorithm", "top_k"),
    ],
)
def test_any_param_change_changes_id(field, new_value):
    changed = {**BASE, field: new_value}
    assert derive_model_id(**changed) != derive_model_id(**BASE)


def test_temperature_int_vs_float_differ():
    """1 (int) and 1.0 (float) serialise differently -> different ids. The
    Pydantic float coercion on the declare path is what protects against this;
    this test documents the trap for anything that bypasses the schema."""
    as_int = derive_model_id(
        model_name="m", temperature=1, top_k=None, top_p=None,
        seed=None, decoding_algorithm=None,
    )
    as_float = derive_model_id(
        model_name="m", temperature=1.0, top_k=None, top_p=None,
        seed=None, decoding_algorithm=None,
    )
    assert as_int != as_float
