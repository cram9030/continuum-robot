"""
Continuous-time LTI Kalman Filter implementation using LQE-designed gains.

This module provides a continuous-time Linear Time-Invariant (LTI) Kalman filter
that uses pre-computed steady-state estimator gains from an LQE design. This is
more efficient than the time-varying Kalman filter when the system reaches
steady-state.
"""

import numpy as np
import warnings
from scipy.integrate import solve_ivp
from .estimator_abstractions import AbstractEstimatorHandler
from .validator_abstractions import IEstimatorValidator
from .linear_estimator_validator import LinearEstimatorValidator


class KalmanFilterLTI(AbstractEstimatorHandler):
    """
    Continuous-time LTI Kalman Filter using steady-state gains.

    This filter implements a continuous-time Linear Time-Invariant Kalman filter
    that uses pre-computed steady-state estimator gains. Unlike time-varying
    Kalman filters that update covariance matrices at each step, this filter
    uses constant gains, making it more efficient for systems at steady-state.

    For continuous-time systems:
        dx̂/dt = A*x̂ + B*u + L*(y - C*x̂)
                = (A - L*C)*x̂ + B*u + L*y

    For discrete-time systems:
        x̂[k+1] = (A - L*C)*x̂[k] + B*u[k] + L*y[k]

    The key difference from hybrid or traditional Kalman filters is that this
    filter does not update the covariance matrix P during operation - it uses
    the steady-state value computed during initialization.

    Attributes:
        A: State transition matrix
        B: Input matrix
        C: Observation/output matrix
        L: Steady-state estimator gain matrix
        P: Steady-state error covariance matrix
        A_est: Pre-computed estimator dynamics matrix (A - L*C)
        x_est: Current state estimate
        dt: Time step (None for continuous-time)
        is_continuous: Whether the system is continuous-time
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        L: np.ndarray,
        P: np.ndarray,
        x0: np.ndarray = None,
        dt: float = None,
        validator: IEstimatorValidator = None,
    ):
        """
        Initialize the LTI Kalman Filter with pre-computed gains.

        Args:
            A: State transition matrix (n_states x n_states)
            B: Input matrix (n_states x n_inputs)
            C: Observation/output matrix (n_measurements x n_states)
            L: Steady-state estimator gain matrix (n_states x n_measurements)
            P: Steady-state error covariance matrix (n_states x n_states)
            x0: Initial state estimate (n_states,). If None, defaults to zeros.
            dt: Time step for discrete-time systems. If None, assumes continuous-time.
            validator: Validation strategy (default: LinearEstimatorValidator)

        Raises:
            ValueError: If validation fails with errors
        """
        # Use default validator if none provided
        if validator is None:
            validator = LinearEstimatorValidator()

        self.validator = validator

        # Store dimensions
        n_states = A.shape[0]

        # Set initial state if not provided
        if x0 is None:
            x0 = np.zeros(n_states)

        # Determine if continuous or discrete
        self.is_continuous = dt is None or dt == 0
        self.dt = dt

        # Handle numerical precision issues with P from Riccati equation solution
        # The algebraic Riccati equation solution should be positive semi-definite,
        # but may have small negative eigenvalues due to numerical precision.
        # Regularize P BEFORE validation to ensure it passes the PSD check.
        P_regularized = P.copy()

        # Ensure P is symmetric (should be from ARE solution, but enforce for safety)
        P_regularized = 0.5 * (P_regularized + P_regularized.T)

        # Check eigenvalues and regularize if needed
        eigenvals = np.linalg.eigvals(P_regularized)
        min_eigenval = np.min(np.real(eigenvals))

        if min_eigenval < -1e-8:
            # Significant negative eigenvalues - this is unexpected
            warnings.warn(
                f"Covariance matrix P has negative eigenvalues (min: {min_eigenval:.2e}). "
                f"This suggests numerical issues in the Riccati equation solution. "
                f"Applying regularization."
            )
            # Eigenvalue decomposition and regularization
            eigenvals_real, eigenvecs = np.linalg.eigh(P_regularized)
            eigenvals_real = np.maximum(
                eigenvals_real, 0
            )  # Zero out negative eigenvalues
            P_regularized = eigenvecs @ np.diag(eigenvals_real) @ eigenvecs.T
        elif min_eigenval < 0:
            # Small negative eigenvalues due to numerical precision - zero them out silently
            eigenvals_real, eigenvecs = np.linalg.eigh(P_regularized)
            eigenvals_real = np.maximum(eigenvals_real, 0)
            P_regularized = eigenvecs @ np.diag(eigenvals_real) @ eigenvecs.T

        # Create dummy Q and R for validation (not used in runtime)
        Q_dummy = np.eye(n_states)
        n_measurements = C.shape[0]
        R_dummy = np.eye(n_measurements)

        # Determine dt for validation
        dt_for_validation = None if self.is_continuous else (dt, dt)

        # Validate initialization parameters (use regularized P)
        validation_result = self.validator.validate_initialization(
            A, B, C, Q_dummy, R_dummy, P_regularized, x0, dt=dt_for_validation
        )

        # Raise errors if validation failed
        if validation_result.has_errors():
            error_messages = "\n".join(validation_result.get_errors())
            raise ValueError(
                f"LTI Kalman filter initialization failed:\n{error_messages}"
            )

        # Issue warnings for non-critical issues
        for warning_msg in validation_result.get_warnings():
            warnings.warn(warning_msg)

        # Validate L dimensions
        if L.ndim != 2:
            raise ValueError(f"Estimator gain L must be 2-dimensional, got {L.ndim}D")
        if L.shape != (n_states, n_measurements):
            raise ValueError(
                f"Estimator gain L must have shape ({n_states}, {n_measurements}), got {L.shape}"
            )

        # Check for NaN/inf in L
        if np.any(np.isnan(L)):
            raise ValueError("Estimator gain L contains NaN values")
        if np.any(np.isinf(L)):
            raise ValueError("Estimator gain L contains infinite values")

        # Store system matrices (make copies to prevent external modification)
        self.A = A.copy()
        self.B = B.copy()
        self.C = C.copy()
        self.L = L.copy()
        self.P = P_regularized  # Use the regularized P matrix
        self.x_est = x0.copy()

        # Pre-compute constant matrices for efficiency
        self.A_est = self.A - self.L @ self.C  # Estimator dynamics matrix

        # Store dimensions for runtime validation
        self.n_states = n_states
        self.n_measurements = n_measurements
        self.n_inputs = B.shape[1] if B.ndim > 1 else 1

        # For continuous-time systems, store the last time for integration
        self.last_time = None

    def estimate_states(self, y: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """
        Estimate current states given observations and input.

        For continuous-time systems, integrates the estimator dynamics from
        the last time to current time. For discrete-time systems, performs
        a single discrete update.

        Args:
            y: Current measurement vector (n_measurements,)
            u: Current input vector (n_inputs,)
            t: Current time

        Returns:
            np.ndarray: Current state vector estimation (n_states,)

        Raises:
            ValueError: If runtime validation fails
        """
        # Validate runtime state
        validation_result = self.validator.validate_runtime_state(
            x=self.x_est,
            y=y,
            u=u,
            n_states=self.n_states,
            n_measurements=self.n_measurements,
            n_inputs=self.n_inputs,
        )

        if validation_result.has_errors():
            error_messages = "\n".join(validation_result.get_errors())
            raise ValueError(f"Runtime validation failed:\n{error_messages}")

        if self.is_continuous:
            # Continuous-time estimator integration
            if self.last_time is None:
                # First call - no integration needed, just apply correction
                self.last_time = t
                innovation = y - self.C @ self.x_est
                self.x_est = self.x_est + self.L @ innovation
            else:
                # Integrate estimator dynamics from last_time to t
                dt_elapsed = t - self.last_time
                if dt_elapsed < 0:
                    # Time went backwards - raise error
                    raise ValueError(
                        f"Time cannot go backwards in continuous-time estimation. "
                        f"Last time: {self.last_time}, current time: {t}"
                    )
                elif dt_elapsed > 0:
                    # Estimator dynamics: dx̂/dt = A_est*x̂ + B*u + L*y
                    # where A_est = A - L*C (pre-computed)

                    # Pre-compute constant terms
                    Bu_plus_Ly = self.B @ u + self.L @ y

                    def estimator_dynamics(t_local, x_local):
                        return self.A_est @ x_local + Bu_plus_Ly

                    # Integrate using solve_ivp
                    sol = solve_ivp(
                        estimator_dynamics,
                        (self.last_time, t),
                        self.x_est,
                        method="RK45",
                        rtol=1e-6,
                        atol=1e-8,
                    )

                    if not sol.success:
                        warnings.warn(f"Integration failed: {sol.message}")
                    else:
                        self.x_est = sol.y[:, -1]

                    self.last_time = t
                # else: dt_elapsed == 0, no integration needed

        else:
            # Discrete-time estimator update
            # x̂[k+1] = A_est*x̂[k] + B*u[k] + L*y[k]
            # where A_est = A - L*C (pre-computed)
            self.x_est = self.A_est @ self.x_est + self.B @ u + self.L @ y

        # Return copy to prevent external modification
        return self.x_est.copy()

    def reset(self, x0: np.ndarray = None):
        """
        Reset the estimator state.

        Args:
            x0: New initial state estimate. If None, resets to zeros.
        """
        if x0 is None:
            self.x_est = np.zeros(self.n_states)
        else:
            if x0.shape != (self.n_states,):
                raise ValueError(
                    f"Initial state x0 must have shape ({self.n_states},), got {x0.shape}"
                )
            self.x_est = x0.copy()

        self.last_time = None
