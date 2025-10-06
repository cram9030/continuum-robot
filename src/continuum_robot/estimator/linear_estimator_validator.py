"""
Linear estimator validation implementation.

This module provides validation for linear Kalman filters, including checks for
matrix dimensions, numerical properties, observability, and stability.
"""

import numpy as np
from .validator_abstractions import IEstimatorValidator, ValidationResult


class LinearEstimatorValidator(IEstimatorValidator):
    """
    Validator for linear Kalman filter estimators.

    Performs comprehensive validation of initialization parameters and runtime states
    for linear time-invariant (LTI) systems with the standard Kalman filter formulation:
        x_k+1 = A @ x_k + B @ u_k + w_k    (w_k ~ N(0, Q))
        y_k = C @ x_k + v_k                 (v_k ~ N(0, R))
    """

    def validate_initialization(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        P: np.ndarray,
        x0: np.ndarray,
        dt: float,
    ) -> ValidationResult:
        """
        Validate initialization parameters for a linear Kalman filter.

        Args:
            A: State transition matrix (n_states x n_states)
            B: Control input matrix (n_states x n_inputs)
            C: Observation matrix (n_measurements x n_states)
            Q: Process noise covariance (n_states x n_states)
            R: Measurement noise covariance (n_measurements x n_measurements)
            P: Initial estimate error covariance (n_states x n_states)
            x0: Initial state estimate (n_states,)
            dt: Time step (must be positive)

        Returns:
            ValidationResult with validation status and messages
        """
        result = ValidationResult()

        # Type validation
        self._validate_types(result, A, B, C, Q, R, P, x0, dt)
        if result.has_errors():
            return result

        # Dimension validation
        n_states = A.shape[0]
        n_measurements = C.shape[0]
        self._validate_dimensions(
            result, A, B, C, Q, R, P, x0, n_states, n_measurements
        )
        if result.has_errors():
            return result

        # Value validation (NaN/inf checks)
        self._validate_values(result, A, B, C, Q, R, P, x0)
        if result.has_errors():
            return result

        # Covariance matrix properties
        self._validate_covariance_matrices(result, Q, R, P)
        if result.has_errors():
            return result

        # System-specific properties (warnings only)
        self._check_observability(result, A, C)
        self._check_stability(result, A)
        self._check_conditioning(result, A, P)
        self._check_time_step(result, dt)
        self._check_system_size(result, n_states)

        return result

    def validate_runtime_state(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        n_states: int,
        n_measurements: int,
        n_inputs: int,
    ) -> ValidationResult:
        """
        Validate runtime state during estimation.

        Args:
            x: Current state estimate (n_states,)
            y: Current measurement (n_measurements,)
            u: Current input (n_inputs,)
            n_states: Expected number of states
            n_measurements: Expected number of measurements
            n_inputs: Expected number of inputs

        Returns:
            ValidationResult with validation status and messages
        """
        result = ValidationResult()

        # Type checks
        if not isinstance(x, np.ndarray):
            result.add_error(f"State x must be numpy array, got {type(x)}")
        if not isinstance(y, np.ndarray):
            result.add_error(f"Measurement y must be numpy array, got {type(y)}")
        if not isinstance(u, np.ndarray):
            result.add_error(f"Input u must be numpy array, got {type(u)}")

        if result.has_errors():
            return result

        # Dimension checks
        if x.shape != (n_states,):
            result.add_error(f"State x must have shape ({n_states},), got {x.shape}")
        if y.shape != (n_measurements,):
            result.add_error(
                f"Measurement y must have shape ({n_measurements},), got {y.shape}"
            )
        if u.shape != (n_inputs,):
            result.add_error(f"Input u must have shape ({n_inputs},), got {u.shape}")

        # Value checks
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            result.add_error("State x contains NaN or infinite values")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            result.add_error("Measurement y contains NaN or infinite values")
        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            result.add_error("Input u contains NaN or infinite values")

        return result

    def _validate_types(
        self,
        result: ValidationResult,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        P: np.ndarray,
        x0: np.ndarray,
        dt: float,
    ) -> None:
        """Validate types of input matrices and dt."""
        matrices = {"A": A, "B": B, "C": C, "Q": Q, "R": R, "P": P, "x0": x0}

        for name, matrix in matrices.items():
            if not isinstance(matrix, np.ndarray):
                result.add_error(
                    f"Matrix {name} must be a numpy array, got {type(matrix)}"
                )
            elif not np.issubdtype(matrix.dtype, np.floating):
                result.add_warning(
                    f"Matrix {name} should be floating point type for numerical "
                    f"stability, got {matrix.dtype}"
                )

        if not isinstance(dt, (int, float)):
            result.add_error(f"dt must be numeric, got {type(dt)}")
        elif dt <= 0:
            result.add_error(f"dt must be positive, got {dt}")

    def _validate_dimensions(
        self,
        result: ValidationResult,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        P: np.ndarray,
        x0: np.ndarray,
        n_states: int,
        n_measurements: int,
    ) -> None:
        """Validate dimensions of system matrices."""
        # Check A is square
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            result.add_error(f"A must be square ({n_states}x{n_states}), got {A.shape}")
            return

        # Check B dimensions
        if B.ndim != 2 or B.shape[0] != n_states:
            result.add_error(f"B must have {n_states} rows to match A, got {B.shape}")

        # Check C dimensions
        if C.ndim != 2 or C.shape[1] != n_states:
            result.add_error(
                f"C must have {n_states} columns to match A, got {C.shape}"
            )

        # Check Q dimensions
        if Q.shape != (n_states, n_states):
            result.add_error(
                f"Q must match A dimensions ({n_states}x{n_states}), got {Q.shape}"
            )

        # Check R dimensions
        if R.shape != (n_measurements, n_measurements):
            result.add_error(
                f"R must be {n_measurements}x{n_measurements}, got {R.shape}"
            )

        # Check P dimensions
        if P.shape != (n_states, n_states):
            result.add_error(
                f"P must match A dimensions ({n_states}x{n_states}), got {P.shape}"
            )

        # Check x0 dimensions
        if x0.shape != (n_states,):
            result.add_error(f"x0 must have {n_states} elements, got {x0.shape}")

        # Check for empty matrices
        if A.size == 0 or B.size == 0 or C.size == 0:
            result.add_error("System matrices cannot be empty")

    def _validate_values(
        self,
        result: ValidationResult,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        P: np.ndarray,
        x0: np.ndarray,
    ) -> None:
        """Validate that matrices don't contain NaN or infinite values."""
        matrices = {"A": A, "B": B, "C": C, "Q": Q, "R": R, "P": P, "x0": x0}

        for name, matrix in matrices.items():
            if np.any(np.isnan(matrix)):
                result.add_error(f"Matrix {name} contains NaN values")
            if np.any(np.isinf(matrix)):
                result.add_error(f"Matrix {name} contains infinite values")

    def _validate_covariance_matrices(
        self, result: ValidationResult, Q: np.ndarray, R: np.ndarray, P: np.ndarray
    ) -> None:
        """Validate covariance matrix properties."""
        if not self._is_positive_semidefinite(Q):
            result.add_error("Process noise covariance Q must be positive semidefinite")

        if not self._is_positive_definite(R):
            result.add_error("Measurement noise covariance R must be positive definite")

        if not self._is_positive_semidefinite(P):
            result.add_error("Initial error covariance P must be positive semidefinite")

    def _check_observability(
        self, result: ValidationResult, A: np.ndarray, C: np.ndarray
    ) -> None:
        """Check system observability using observability matrix rank test."""
        try:
            n_states = A.shape[0]
            # Construct observability matrix O = [C; C*A; C*A^2; ...; C*A^(n-1)]
            obs_matrix = np.vstack(
                [C @ np.linalg.matrix_power(A, i) for i in range(n_states)]
            )
            rank = np.linalg.matrix_rank(obs_matrix)

            if rank < n_states:
                result.add_warning(
                    f"System may not be fully observable: observability matrix "
                    f"rank is {rank}, expected {n_states}. Some states may not "
                    f"be estimable from measurements."
                )
            else:
                result.add_info("System is fully observable")
        except Exception as e:
            result.add_warning(
                f"Could not check observability due to numerical issues: {e}"
            )

    def _check_stability(self, result: ValidationResult, A: np.ndarray) -> None:
        """Check eigenvalues for stability."""
        try:
            eigenvals = np.linalg.eigvals(A)
            max_real_part = np.max(np.real(eigenvals))

            if max_real_part > 0:
                result.add_warning(
                    f"System matrix A may be unstable (max eigenvalue real part: "
                    f"{max_real_part:.6f}). For discrete-time systems, check if "
                    f"max(abs(eigenvalues)) > 1."
                )
        except Exception as e:
            result.add_warning(f"Could not check stability: {e}")

    def _check_conditioning(
        self, result: ValidationResult, A: np.ndarray, P: np.ndarray
    ) -> None:
        """Check condition numbers for numerical issues."""
        try:
            cond_A = np.linalg.cond(A)
            if cond_A > 1e12:
                result.add_warning(
                    f"System matrix A is ill-conditioned (condition number: "
                    f"{cond_A:.2e}). This may cause numerical instability."
                )

            cond_P = np.linalg.cond(P)
            if cond_P > 1e12:
                result.add_warning(
                    f"Initial covariance P is ill-conditioned (condition number: "
                    f"{cond_P:.2e}). This may cause numerical instability."
                )
        except Exception as e:
            result.add_warning(f"Could not check conditioning: {e}")

    def _check_time_step(self, result: ValidationResult, dt: float) -> None:
        """Check for extremely large time steps."""
        if dt > 1.0:
            result.add_warning(
                f"Large time step dt={dt}s may cause numerical instability in "
                f"discrete-time integration"
            )

    def _check_system_size(self, result: ValidationResult, n_states: int) -> None:
        """Check if state dimension suggests position/velocity structure."""
        if n_states % 2 == 0:
            n_dof = n_states // 2
            if n_dof > 50:
                result.add_warning(
                    f"Large number of DOFs ({n_dof}) detected. Consider model "
                    f"reduction for real-time applications."
                )

    @staticmethod
    def _is_positive_definite(matrix: np.ndarray) -> bool:
        """Check if matrix is positive definite using Cholesky decomposition."""
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def _is_positive_semidefinite(matrix: np.ndarray) -> bool:
        """Check if matrix is positive semidefinite using eigenvalues."""
        try:
            eigenvals = np.linalg.eigvals(matrix)
            return np.all(eigenvals >= -1e-8)  # Small tolerance for numerical errors
        except Exception:
            return False
