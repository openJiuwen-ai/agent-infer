"""Validate the eval_result JSON Schema against a sample, and reject bad ones."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

REPO = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO / "schemas" / "eval_result.schema.json"
SAMPLE_PATH = Path(__file__).parent / "fixtures" / "eval_result_sample.json"


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture()
def schema():
    return _load(SCHEMA_PATH)


@pytest.fixture()
def sample():
    return _load(SAMPLE_PATH)


def test_sample_is_valid(schema, sample):
    jsonschema.validate(instance=sample, schema=schema)


def test_missing_required_field_rejected(schema, sample):
    bad = copy.deepcopy(sample)
    del bad["outcome_class"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_bad_outcome_class_rejected(schema, sample):
    bad = copy.deepcopy(sample)
    bad["outcome_class"] = "totally_made_up"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_source_must_be_real_vllm(schema, sample):
    bad = copy.deepcopy(sample)
    bad["source"] = "sim"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_empty_raw_per_seed_rejected(schema, sample):
    bad = copy.deepcopy(sample)
    bad["raw_per_seed_metrics"] = []
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)
