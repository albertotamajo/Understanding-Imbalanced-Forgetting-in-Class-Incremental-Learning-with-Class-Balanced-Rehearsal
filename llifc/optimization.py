from __future__ import annotations

from numbers import Integral
from typing import Any

import cvxpy as cp
import numpy as np
import numpy.typing as npt


class LastLayerBiasInterferenceMinimizer:
    """
    Reusable CVXPY model for minimizing last-layer bias interference.

    Construct once for fixed row count I, class count C, and feature dimension d,
    then call solve() for each independent (F_n, S_n, H_n, V_n) problem. The
    positive normalizer m_n is supplied to solve(); it need not equal I.

    With G = (S_prime - H).T @ F / m, minimize

        -<G - V, V>_F / ||V||_F + lambda_reg * ||G - V||_F**2

    subject to row sums of one, S_prime >= delta, sum(KL(S_prime || S)) / m
    <= epsilon, interference >= eta, and ||G||_F**2 <= ||V||_F**2.
    Here interference = -<G - V, V>_F / ||V||_F, and lambda_reg >= 0 is
    supplied to solve(). Its default of zero recovers the unregularized problem.
    Optimization is skipped when the initial interference is below eta.

    CVXPY does not support strict inequalities. A positive delta implements
    S_prime > 0 numerically, restricting the feasible set to S_prime >= delta.
    Constraints hold to the selected solver's tolerances; the returned
    diagnostics describe the unmodified solution, without clipping probabilities.

    Products involving input data are evaluated in NumPy so the model obeys
    disciplined parametrized programming (DPP) and can reuse its compilation.
    """

    def __init__(
        self,
        I: int,
        C: int,
        d: int,
        delta: float = 1e-8,
    ) -> None:
        if any(
            not isinstance(size, Integral) or isinstance(size, bool) or size <= 0
            for size in (I, C, d)
        ):
            raise ValueError("I, C, and d must be positive integers.")

        if not np.isfinite(delta) or not 0 < delta < 1 / C:
            raise ValueError("delta must be finite and satisfy 0 < delta < 1/C.")

        self.I = int(I)
        self.C = int(C)
        self.d = int(d)
        self.delta = float(delta)
        self.n_variables = self.I * self.C

        # Q_over_m = F @ (V / ||V||_F).T / m, so interference is affine:
        # -<S_prime, Q_over_m>_F + <H, Q_over_m>_F + ||V||_F.
        self.Q_over_m = cp.Parameter((I, C), name="Q_over_m")
        self.interference_offset = cp.Parameter(name="interference_offset")

        # Precompute H.T @ F / m rather than multiplying two parameters.
        self.F_over_m = cp.Parameter((I, d), name="F_over_m")
        self.HF_over_m = cp.Parameter((C, d), name="HF_over_m")
        self.V_norm_squared = cp.Parameter(nonneg=True, name="V_norm_squared")

        # Scale the residual's coefficients by sqrt(lambda_reg) in NumPy to
        # keep the quadratic regularizer DPP even when lambda_reg changes.
        self.sqrt_lambda_F_over_m = cp.Parameter((I, d), name="sqrt_lambda_F_over_m")
        self.regularization_offset = cp.Parameter((C, d), name="regularization_offset")

        self.log_S = cp.Parameter((I, C), name="log_S")
        # Multiply epsilon by m outside CVXPY, avoiding division by a parameter.
        self.kl_budget = cp.Parameter(nonneg=True, name="kl_budget")
        self.eta = cp.Parameter(name="eta")

        self.S_prime = cp.Variable((I, C), name="S_prime")
        self.interference = (
            -cp.sum(cp.multiply(self.S_prime, self.Q_over_m))
            + self.interference_offset
        )
        self.total_kl = (
            -cp.sum(cp.entr(self.S_prime))
            - cp.sum(cp.multiply(self.S_prime, self.log_S))
        )
        self.gradient = self.S_prime.T @ self.F_over_m - self.HF_over_m
        self.gradient_norm_squared = cp.sum_squares(self.gradient)
        self.regularization = cp.sum_squares(
            self.S_prime.T @ self.sqrt_lambda_F_over_m - self.regularization_offset
        )

        # Keep all constant terms so optimal_value is the full objective.
        self.objective = cp.Minimize(self.interference + self.regularization)
        self.constraints = [
            cp.sum(self.S_prime, axis=1) == 1,
            self.S_prime >= self.delta,
            self.total_kl <= self.kl_budget,
            self.interference >= self.eta,
            self.gradient_norm_squared <= self.V_norm_squared,
        ]
        self.problem = cp.Problem(self.objective, self.constraints)

    def solve(
        self,
        F: npt.ArrayLike,
        S: npt.ArrayLike,
        H: npt.ArrayLike,
        V: npt.ArrayLike,
        m: float,
        epsilon: float,
        eta: float,
        solver: str | None = None,
        canon_backend: str | None = None,
        *,
        lambda_reg: float = 0.0,
        **solver_options: Any,
    ) -> dict[str, Any]:
        """
        Solve one independent problem using float64 NumPy data.

        F: shape (I, d).
        S: shape (I, C), finite, strictly positive, with each row summing to one.
        H: shape (I, C), typically one-hot target encodings.
        V: shape (C, d), with a finite, nonzero Frobenius norm.
        m: finite positive normalizer, independent of the row count I.
        epsilon: finite nonnegative KL budget after division by m.
        eta: finite lower bound on the final bias interference.
        solver: an exponential-cone solver; None lets CVXPY choose an installed
            solver. Additional options are forwarded to problem.solve().
        lambda_reg: finite nonnegative weight on ||G - V||_F**2. Zero disables
            regularization.

        Return a dictionary in the style of ForgettingBalancer.solve(), with
        status, S_prime, the full optimal objective, timings, and constraint
        diagnostics. average_kl is sum_i KL(S_prime[i] || S[i]) / m.
        optimal_value includes interference + regularization, where
        regularization = lambda_reg * gradient_residual_norm_squared.
        interference remains the unregularized bias interference.
        Constraint violations are nonnegative excesses, zero when satisfied.

        If initial_interference < eta, return an unchanged copy of S with
        status="skipped" and optimal_value=None. This is a policy decision,
        not a claim that the constrained problem is infeasible; the returned S
        need not satisfy the interference or norm constraints.

        An eligible input need not satisfy the norm constraint initially. The
        solver may repair it, or report infeasibility. For a status other than
        optimal/optimal_inaccurate, S_prime and solution diagnostics are None.
        Solver errors propagate as CVXPY exceptions.
        """
        F = np.asarray(F, dtype=np.float64)
        S = np.asarray(S, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)

        for name, value, shape in (
            ("F", F, (self.I, self.d)),
            ("S", S, (self.I, self.C)),
            ("H", H, (self.I, self.C)),
            ("V", V, (self.C, self.d)),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}.")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Every entry of {name} must be finite.")

        if np.any(S <= 0):
            raise ValueError("Every entry of S must be positive.")
        if not np.allclose(S.sum(axis=1), 1, rtol=0, atol=1e-7):
            raise ValueError("Each row of S must sum to one.")

        for name, value in (
            ("m", m), ("epsilon", epsilon), ("eta", eta), ("lambda_reg", lambda_reg)
        ):
            if np.ndim(value) != 0 or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar.")
        m, epsilon, eta = float(m), float(epsilon), float(eta)
        lambda_reg = float(lambda_reg)
        if m <= 0:
            raise ValueError("m must be positive.")
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative.")
        if lambda_reg < 0:
            raise ValueError("lambda_reg must be nonnegative.")

        V_norm = float(np.linalg.norm(V))
        if not np.isfinite(V_norm) or V_norm == 0:
            raise ValueError("V must have a finite nonzero Frobenius norm.")

        Q_over_m = (F @ (V / V_norm).T) / m
        initial_interference = float(V_norm - np.sum((S - H) * Q_over_m))
        if not np.isfinite(initial_interference):
            raise ValueError("The initial interference must be finite.")
        eligible = initial_interference >= eta

        result: dict[str, Any] = {
            "status": "skipped",
            "eligible": eligible,
            "initial_interference": initial_interference,
            "V_norm": V_norm,
            "optimal_value": None,
            "compilation_time": 0.0,
            "solver_time": 0.0,
            "iterations": 0,
            "S_prime": None,
            "interference": None,
            "gradient_residual_norm_squared": None,
            "regularization": None,
            "average_kl": None,
            "gradient_norm_squared": None,
            "row_sum_error": None,
            "minimum_probability": None,
            "kl_constraint_violation": None,
            "interference_constraint_violation": None,
            "norm_constraint_violation": None,
            "probability_constraint_violation": None,
        }

        if eligible:
            self.Q_over_m.value = Q_over_m
            self.interference_offset.value = float(np.sum(H * Q_over_m) + V_norm)
            self.F_over_m.value = F / m
            self.HF_over_m.value = H.T @ (F / m)
            self.V_norm_squared.value = V_norm * V_norm
            sqrt_lambda = np.sqrt(lambda_reg)
            self.sqrt_lambda_F_over_m.value = sqrt_lambda * self.F_over_m.value
            self.regularization_offset.value = sqrt_lambda * (self.HF_over_m.value + V)
            self.log_S.value = np.log(S)
            self.kl_budget.value = m * epsilon
            self.eta.value = eta

            solve_kwargs: dict[str, Any] = {
                "solver": solver,
                "warm_start": True,
                "verbose": False,
                **solver_options,
            }
            if canon_backend is not None:
                solve_kwargs["canon_backend"] = canon_backend

            self.problem.solve(**solve_kwargs)

            result.update(
                {
                    "status": self.problem.status,
                    "optimal_value": self.problem.value,
                    "compilation_time": self.problem._compilation_time,
                    "solver_time": self.problem.solver_stats.solve_time,
                    "iterations": self.problem.solver_stats.num_iters,
                }
            )
            if (
                self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                or self.S_prime.value is None
            ):
                return result

            S_value = np.array(self.S_prime.value, copy=True)
        else:
            S_value = S.copy()

        # Recompute diagnostics from the returned matrix using the definitions.
        gradient = (S_value - H).T @ (F / m)
        interference = float(-np.sum((gradient - V) * (V / V_norm)))
        gradient_norm_squared = float(np.sum(gradient**2))
        gradient_residual_norm_squared = float(np.sum((gradient - V)**2))
        if np.any(S_value < 0):
            average_kl = float("inf")
        else:
            positive = S_value > 0
            average_kl = float(np.sum(
                S_value[positive] * (np.log(S_value[positive]) - np.log(S[positive]))
            ) / m)

        result.update(
            {
                "S_prime": S_value,
                "interference": interference,
                "gradient_residual_norm_squared": gradient_residual_norm_squared,
                "regularization": lambda_reg * gradient_residual_norm_squared,
                "average_kl": average_kl,
                "gradient_norm_squared": gradient_norm_squared,
                "row_sum_error": float(np.max(np.abs(S_value.sum(axis=1) - 1))),
                "minimum_probability": float(np.min(S_value)),
                "kl_constraint_violation": max(0.0, average_kl - epsilon),
                "interference_constraint_violation": max(0.0, eta - interference),
                "norm_constraint_violation": max(0.0, gradient_norm_squared - V_norm**2),
                "probability_constraint_violation": max(0.0, self.delta - float(S_value.min())),
            }
        )
        return result
