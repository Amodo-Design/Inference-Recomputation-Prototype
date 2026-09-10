"""Pure tests for the Base64Bytes codec (app/schemas.py).

Isolates the base64<->bytes handling from the DB round-trip tests, and pins the
integrity guarantee that malformed base64 is rejected (not silently stripped).
"""

from __future__ import annotations

import base64

import pytest
from pydantic import BaseModel, ValidationError

from app.schemas import Base64Bytes


class Model(BaseModel):
    data: Base64Bytes


class OptModel(BaseModel):
    data: Base64Bytes | None = None


def test_base64_string_decodes_to_bytes():
    m = Model(data=base64.b64encode(b"\x00\x01\x02hello").decode())
    assert m.data == b"\x00\x01\x02hello"


def test_json_dump_reencodes_to_base64():
    raw = b"\xff\xfe\x00binary"
    m = Model(data=base64.b64encode(raw).decode())
    dumped = m.model_dump(mode="json")["data"]
    assert isinstance(dumped, str)
    assert base64.b64decode(dumped) == raw


def test_python_mode_keeps_bytes():
    # Serializer is when_used="json", so python-mode dump stays bytes.
    m = Model(data=base64.b64encode(b"abc").decode())
    assert m.model_dump()["data"] == b"abc"


def test_raw_bytes_input_passes_through():
    # Mirrors reading a value straight off the ORM (already bytes).
    m = Model(data=b"raw-bytes")
    assert m.data == b"raw-bytes"


def test_optional_none_passthrough():
    m = OptModel()
    assert m.data is None
    assert m.model_dump(mode="json")["data"] is None


@pytest.mark.parametrize("bad", ["@@@@invalid@@@@", "abc", "not base64!"])
def test_malformed_base64_rejected(bad):
    with pytest.raises(ValidationError):
        Model(data=bad)
