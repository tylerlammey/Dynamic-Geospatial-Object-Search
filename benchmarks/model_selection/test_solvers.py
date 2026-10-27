"""CPU numerical/dispatch tests; GPU outputs must be separately validated."""

from __future__ import annotations

import unittest

import torch
from solvers import neumann_triangular_solve


class ExperimentalSolverTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261026)
        self.native = torch.linalg.solve_triangular

    def assert_matches(self, A, B, **context):
        expected = self.native(A, B, upper=False, unitriangular=True)
        with neumann_triangular_solve(**context) as solver:
            actual = torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        return solver.stats

    def test_normalized_model_like_keys_multiple_batches_heads_chunks_and_rhs(self):
        # Qwen's A=(beta_i * dot(k_i,k_j))*exp(cumulative_g_i-cumulative_g_j),
        # with an implicit unit diagonal. Its consecutive RHS are V_beta and
        # decayed K_beta. These shapes include batches, heads and padded chunks.
        keys = torch.nn.functional.normalize(torch.randn(2, 3, 4, 64, 32), dim=-1)
        beta = torch.rand(2, 3, 4, 64, 1) * 0.9 + 0.05
        cumulative = -(torch.rand(2, 3, 4, 64) * 0.3 + 0.01).cumsum(-1)
        decay = (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).tril().exp()
        A = ((keys * beta) @ keys.transpose(-1, -2) * decay).tril()
        rhs_values = torch.randn(2, 3, 4, 64, 24) * beta
        rhs_keys = keys * beta * cumulative.exp().unsqueeze(-1)
        expected_values = self.native(A, rhs_values, upper=False, unitriangular=True)
        expected_keys = self.native(A, rhs_keys, upper=False, unitriangular=True)
        with neumann_triangular_solve() as solver:
            values = torch.linalg.solve_triangular(A, rhs_values, upper=False, unitriangular=True)
            decayed_keys = torch.linalg.solve_triangular(
                A, rhs_keys, upper=False, unitriangular=True
            )
        torch.testing.assert_close(values, expected_values, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(decayed_keys, expected_keys, atol=1e-4, rtol=1e-4)
        self.assertEqual(solver.stats["optimized_calls"], 2)
        self.assertEqual(solver.stats["inverse_builds"], 1)
        self.assertEqual(solver.stats["inverse_cache_hits"], 1)

    def test_random_triangular_orders_and_broadcast_rhs(self):
        for order in (1, 2, 4, 8, 16, 32, 64):
            with self.subTest(order=order):
                A = torch.randn(2, 3, order, order) * 0.03
                B = torch.randn(3, order, 7)
                stats = self.assert_matches(A, B)
                self.assertEqual(stats["optimized_calls"], 1)

    def test_zero_lower_triangle_ignores_diagonal_and_upper_values(self):
        A = torch.zeros(64, 64)
        A.diagonal().fill_(float("nan"))
        A[0, 63] = float("inf")
        stats = self.assert_matches(A, torch.randn(64, 8))
        self.assertEqual(stats["optimized_calls"], 1)

    def test_correlated_keys_reject_unstable_neumann_intermediates(self):
        # All normalized keys coincide; beta approaches one and decay approaches
        # zero. The true inverse is manageable but N**32 suffers cancellation.
        A = torch.full((2, 64, 64), 0.99)
        B = torch.randn(2, 64, 16)
        stats = self.assert_matches(A, B)
        self.assertEqual(stats["rejected_inverses"], 1)
        self.assertEqual(stats["fallback_calls"], 1)

    def test_inplace_mutation_invalidates_versioned_cache(self):
        A = torch.randn(64, 64) * 0.01
        B = torch.randn(64, 12)
        with neumann_triangular_solve() as solver:
            torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True)
            A[30, 2] += 0.1
            actual = torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True)
        expected = self.native(A, B, upper=False, unitriangular=True)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(solver.stats["inverse_builds"], 2)
        self.assertEqual(solver.stats["inverse_cache_hits"], 0)

    def test_cache_is_consumed_by_second_rhs_and_cleared_at_context_exit(self):
        A, B = torch.randn(64, 64) * 0.01, torch.randn(64, 8)
        with torch.inference_mode(), neumann_triangular_solve() as solver:
            # Newly created tensors have no version counter in this mode.
            A = A.clone()
            for _ in range(3):
                result = torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True)
            self.assertEqual(solver.stats["inverse_builds"], 2)
            self.assertEqual(solver.stats["inverse_cache_hits"], 1)
        self.assertIsNone(solver._cache)
        self.assertFalse(solver.active)
        self.assertIs(torch.linalg.solve_triangular, self.native)
        torch.testing.assert_close(
            result, self.native(A, B, upper=False, unitriangular=True), atol=1e-4, rtol=1e-4
        )

    def test_native_fallback_for_nonunit_upper_right_dtype_nonpower_and_out(self):
        cases = [
            (torch.eye(4) * 2, torch.randn(4, 2), {"upper": False}),
            (torch.eye(4), torch.randn(4, 2), {"upper": True, "unitriangular": True}),
            (
                torch.eye(4),
                torch.randn(2, 4),
                {"upper": False, "left": False, "unitriangular": True},
            ),
            (
                torch.eye(4).double(),
                torch.randn(4, 2).double(),
                {"upper": False, "unitriangular": True},
            ),
            (torch.eye(3), torch.randn(3, 2), {"upper": False, "unitriangular": True}),
        ]
        with neumann_triangular_solve() as solver:
            for A, B, kwargs in cases:
                torch.testing.assert_close(
                    torch.linalg.solve_triangular(A, B, **kwargs), self.native(A, B, **kwargs)
                )
            A, B = torch.eye(4), torch.randn(4, 2)
            out = torch.empty_like(B)
            actual = torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True, out=out)
            self.assertIs(actual, out)
            torch.testing.assert_close(actual, B)
        self.assertEqual(solver.stats["fallback_calls"], 6)
        self.assertEqual(solver.stats["inverse_builds"], 0)

    def test_context_restores_original_after_failure_and_rejects_nesting(self):
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with neumann_triangular_solve():
                raise RuntimeError("deliberate")
        self.assertIs(torch.linalg.solve_triangular, self.native)
        with neumann_triangular_solve():
            with self.assertRaisesRegex(RuntimeError, "Nested"):
                with neumann_triangular_solve():
                    pass
        self.assertIs(torch.linalg.solve_triangular, self.native)


if __name__ == "__main__":
    unittest.main()
