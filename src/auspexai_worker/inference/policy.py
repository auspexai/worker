"""GenerationPolicy — the manifest-declared generation policy the broker
enforces per-request (v0.2 M1 generation-policy half;
inference_determinism_scoping_memo.md §3a/b, RATIFIED 2026-07-02).

Two modes, keyed on the declared temperature:

- **greedy** — temperature 0 (or the `inference_determinism` block omitted):
  the pre-M1 behavior, byte-for-byte. The default; every existing experiment
  is this, unchanged.
- **seeded-sampling** — temperature > 0 with a **pinned seed** (required: the
  reproducibility floor — unseeded sampling is refused). The optional knobs
  `top_p`/`top_k`/`min_p` are a fixed, explicitly-enumerated whitelist
  (memo Q2); the broker honors the DECLARED values and rejects a request
  beyond them.

The policy is parsed from the hash-pinned manifest the provisioning resolver
verified — the same trust root as the serving_version_pin gate. Parse errors
are a REFUSAL (retryable; the unit re-offers elsewhere), never a downgrade to
greedy: silently running a different policy than declared is the
declared-vs-actual lie the coherence gate exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass


class GenerationPolicyError(Exception):
    """The manifest's inference_determinism block cannot be honored as declared.
    Dispatch maps this to a refusal (refuse-don't-echo, refuse-don't-downgrade)."""


@dataclass(frozen=True)
class GenerationPolicy:
    """The declared generation policy for one unit's inference session."""

    temperature: float = 0.0
    seed: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None

    @property
    def is_sampling(self) -> bool:
        return self.temperature > 0

    def knobs(self) -> dict[str, float | int]:
        """The declared sampling knobs, by wire-option name (empty under greedy)."""
        return {k: v for k in ("top_p", "top_k", "min_p") if (v := getattr(self, k)) is not None}

    @classmethod
    def from_manifest(cls, manifest: dict | None) -> GenerationPolicy:
        """Parse + validate the policy from a verified manifest dict.

        Raises GenerationPolicyError on a block this worker cannot honor as
        declared (malformed values, sampling without a pinned seed, knobs under
        greedy). An absent/empty block is the greedy default.
        """
        det = (manifest or {}).get("inference_determinism")
        if det is None:
            return cls()
        if not isinstance(det, dict):
            raise GenerationPolicyError("inference_determinism must be an object")

        raw_temp = det.get("temperature", 0)
        if isinstance(raw_temp, bool) or not isinstance(raw_temp, (int, float)):
            raise GenerationPolicyError("inference_determinism.temperature must be a number")
        temperature = float(raw_temp)
        if temperature < 0:
            raise GenerationPolicyError("inference_determinism.temperature must be >= 0")

        seed = det.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise GenerationPolicyError("inference_determinism.seed must be an integer")

        top_p = det.get("top_p")
        if top_p is not None and (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not (0 < float(top_p) <= 1)
        ):
            raise GenerationPolicyError("inference_determinism.top_p must be in (0, 1]")
        top_k = det.get("top_k")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
        ):
            raise GenerationPolicyError("inference_determinism.top_k must be a positive integer")
        min_p = det.get("min_p")
        if min_p is not None and (
            isinstance(min_p, bool)
            or not isinstance(min_p, (int, float))
            or not (0 <= float(min_p) < 1)
        ):
            raise GenerationPolicyError("inference_determinism.min_p must be in [0, 1)")

        policy = cls(
            temperature=temperature,
            seed=seed,
            top_p=float(top_p) if top_p is not None else None,
            top_k=top_k,
            min_p=float(min_p) if min_p is not None else None,
        )
        if policy.is_sampling and policy.seed is None:
            raise GenerationPolicyError(
                "seeded sampling (temperature > 0) requires a pinned seed "
                "(reproducibility floor — unseeded sampling is refused)"
            )
        if not policy.is_sampling and policy.knobs():
            raise GenerationPolicyError(
                f"sampling knobs {sorted(policy.knobs())} require temperature > 0 "
                "(greedy decoding never reads them)"
            )
        return policy
