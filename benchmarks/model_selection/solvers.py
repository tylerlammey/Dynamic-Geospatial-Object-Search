"""Experimental benchmark-only replacement for tiny FP32 unit-lower TRSM.

This is not a production/model-adapter default. Compare complete warmed GPU
outputs and decision probabilities against native TRSM before interpreting its
speed. CPU numerical tests cannot establish GPU agreement or aerial accuracy.

For a unit-lower matrix A=I-N of power-of-two order n, N**n=0 and
(I-N)**-1 = (I+N)(I+N**2)...(I+N**(n/2)). Five doubling steps suffice for n=64.
All operations below remain FP32. Correlated keys can cause cancellation in
the product, so a finite inverse-residual guard rejects unstable inverses and
uses the unchanged native solver. The guard requires a scalar synchronization
on GPU; its overhead must be included in any timing.

The optional cache holds one strong reference and reuses a checked inverse for
only the immediately following call with the exact same A object. This models
Qwen3.5's two consecutive RHS solves. Ordinary tensor version changes invalidate
the cache. Inference tensors have no version counter: callers explicitly using
the cache must not mutate A between that pair. Always scope the context to one
forward; it clears the cache and restores the original function on every exit.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch


def neumann_inverse(unit_lower: torch.Tensor) -> torch.Tensor:
    """Finite doubling product; caller checks shape, FP32 and numerical residual."""
    order = unit_lower.shape[-1]
    identity = torch.eye(order, device=unit_lower.device, dtype=torch.float32)
    power = -unit_lower.tril(-1)
    inverse = identity + power
    for _ in range(order.bit_length() - 2):
        power = power @ power
        inverse = inverse + inverse @ power
    return inverse


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:  # torch.inference_mode tensors deliberately have no counter.
        return None


class ExperimentalTriangularSolve:
    """Narrow torch.linalg.solve_triangular-compatible wrapper with native fallback."""

    def __init__(self, native, *, cache_two_rhs=True, max_order=64, residual_tolerance=1e-5):
        if max_order < 1 or not math.isfinite(residual_tolerance) or residual_tolerance <= 0:
            raise ValueError("Positive maximum order and finite residual tolerance required")
        self.native = native
        self.cache_two_rhs = cache_two_rhs
        self.max_order = max_order
        self.residual_tolerance = residual_tolerance
        self.active = True
        self._cache = None
        self.stats = {
            "calls": 0,
            "optimized_calls": 0,
            "fallback_calls": 0,
            "inverse_builds": 0,
            "inverse_cache_hits": 0,
            "rejected_inverse_cache_hits": 0,
            "rejected_inverses": 0,
            "maximum_checked_inverse_residual": 0.0,
        }

    def _eligible(self, A, B, upper, left, unitriangular, out):
        if (
            upper
            or not left
            or not unitriangular
            or out is not None
            or not isinstance(A, torch.Tensor)
            or not isinstance(B, torch.Tensor)
            or A.dtype != torch.float32
            or B.dtype != torch.float32
            or A.requires_grad
            or B.requires_grad
            or A.device != B.device
            or A.ndim < 2
            or B.ndim < 2
        ):
            return False
        order = A.shape[-1]
        return (
            0 < order <= self.max_order
            and order & (order - 1) == 0
            and A.shape[-2] == order
            and B.shape[-2] == order
        )

    def __call__(self, A, B, *, upper, left=True, unitriangular=False, out=None):
        if not self.active:
            return self.native(A, B, upper=upper, left=left, unitriangular=unitriangular, out=out)
        self.stats["calls"] += 1
        if not self._eligible(A, B, upper, left, unitriangular, out):
            self._cache = None
            self.stats["fallback_calls"] += 1
            return self.native(A, B, upper=upper, left=left, unitriangular=unitriangular, out=out)
        version = _version(A)
        cached = self._cache
        self._cache = None  # A matching inverse is consumed once; never retained across forwards.
        if cached is not None and cached[0] is A and cached[1] == version:
            inverse = cached[2]
            key = "inverse_cache_hits" if inverse is not None else "rejected_inverse_cache_hits"
            self.stats[key] += 1
        else:
            self.stats["inverse_builds"] += 1
            inverse = neumann_inverse(A)
            identity = torch.eye(A.shape[-1], device=A.device, dtype=torch.float32)
            residual = ((A.tril(-1) + identity) @ inverse - identity).abs().amax().item()
            if math.isfinite(residual):
                self.stats["maximum_checked_inverse_residual"] = max(
                    self.stats["maximum_checked_inverse_residual"], residual
                )
            if not math.isfinite(residual) or residual > self.residual_tolerance:
                inverse = None
                self.stats["rejected_inverses"] += 1
            if self.cache_two_rhs:
                self._cache = (A, version, inverse)
        if inverse is None:
            self.stats["fallback_calls"] += 1
            return self.native(A, B, upper=upper, left=left, unitriangular=unitriangular, out=out)
        self.stats["optimized_calls"] += 1
        return inverse @ B

    def close(self):
        self._cache = None
        self.active = False


@contextmanager
def neumann_triangular_solve(*, cache_two_rhs=True, max_order=64, residual_tolerance=1e-5):
    """Patch TRSM for one serialized forward; restore it even when forward raises.

    Example (timing must include the context and the guard's synchronization)::

        with neumann_triangular_solve() as solver:
            probabilities = adapter.forward(batch)

    This process-global patch must not overlap another forward or a second
    solver context. It never changes the underlying library/model files.
    """
    original = torch.linalg.solve_triangular
    if isinstance(original, ExperimentalTriangularSolve):
        raise RuntimeError("Nested experimental solver contexts are not supported")
    replacement = ExperimentalTriangularSolve(
        original,
        cache_two_rhs=cache_two_rhs,
        max_order=max_order,
        residual_tolerance=residual_tolerance,
    )
    torch.linalg.solve_triangular = replacement
    try:
        yield replacement
    finally:
        torch.linalg.solve_triangular = original
        replacement.close()
