"""v0.2 M1 — GenerationPolicy.from_manifest: parse + validate the declared
generation policy from the hash-pinned manifest. A block the worker cannot
honor AS DECLARED raises (dispatch maps it to a retryable refusal) — never a
silent downgrade to greedy."""

from __future__ import annotations

import pytest

from auspexai_worker.inference.policy import GenerationPolicy, GenerationPolicyError


def test_absent_block_is_greedy():
    for manifest in (None, {}, {"inference_determinism": None}):
        pol = GenerationPolicy.from_manifest(manifest)
        assert not pol.is_sampling
        assert pol.temperature == 0.0
        assert pol.knobs() == {}


def test_greedy_block_parses():
    pol = GenerationPolicy.from_manifest(
        {"inference_determinism": {"temperature": 0, "seed": 7, "serving_version_pin": "x"}}
    )
    assert not pol.is_sampling
    assert pol.seed == 7


def test_sampling_block_parses_with_knobs():
    pol = GenerationPolicy.from_manifest(
        {
            "inference_determinism": {
                "temperature": 0.8,
                "seed": 42,
                "top_p": 0.9,
                "top_k": 40,
                "min_p": 0.05,
            }
        }
    )
    assert pol.is_sampling
    assert pol.knobs() == {"top_p": 0.9, "top_k": 40, "min_p": 0.05}


def test_sampling_without_seed_refused():
    with pytest.raises(GenerationPolicyError, match="pinned seed"):
        GenerationPolicy.from_manifest({"inference_determinism": {"temperature": 0.8}})


def test_knobs_under_greedy_refused():
    with pytest.raises(GenerationPolicyError, match="require temperature > 0"):
        GenerationPolicy.from_manifest({"inference_determinism": {"temperature": 0, "top_p": 0.9}})


@pytest.mark.parametrize(
    "block",
    [
        {"temperature": "hot"},
        {"temperature": -0.1},
        {"temperature": True},
        {"temperature": 0.5, "seed": "x"},
        {"temperature": 0.5, "seed": True},
        {"temperature": 0.5, "seed": 1, "top_p": 0},
        {"temperature": 0.5, "seed": 1, "top_p": 1.5},
        {"temperature": 0.5, "seed": 1, "top_k": 0},
        {"temperature": 0.5, "seed": 1, "top_k": 2.5},
        {"temperature": 0.5, "seed": 1, "min_p": 1.0},
        "not-a-dict",
    ],
)
def test_malformed_blocks_refused(block):
    with pytest.raises(GenerationPolicyError):
        GenerationPolicy.from_manifest({"inference_determinism": block})
