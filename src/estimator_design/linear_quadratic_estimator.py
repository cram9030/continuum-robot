"""
Linear Quadratic Estimator (LQE) design for continuum robot beams.

This module provides the Linear Quadratic Estimator (also known as Kalman Filter
gain design) for linear time-invariant systems. The LQE computes the optimal
estimator gain L that minimizes the steady-state estimation error covariance.
"""

import numpy as np
import control as ct
from continuum_robot.estimator.linear_estimator_validator import (
    LinearEstimatorValidator,
)
from continuum_robot.estimator.validator_abstractions import IEstimatorValidator


class LinearQuadraticEstimator:
    """
    Linear Quadratic Estimator (LQE) for optimal state estimation.

    This class computes the optimal estimator gain matrix L for linear systems
    using the Linear Quadratic Estimator method (also known as Kalman filter design).
    The LQE is the dual of the LQR problem and minimizes the steady-state estimation
    error covariance.

    The estimator gain L is computed by solving the dual algebraic Riccati equation.
    For continuous-time systems:
        dx̂/dt = A*x̂ + B*u + L*(y - C*x̂)

    For discrete-time systems:
        x̂[k+1] = A*x̂[k] + B*u[k] + L*(y[k] - C*x̂[k])

    Attributes:
        A: State transition matrix
        B: Input matrix (used for validation)
        C: Observation/output matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        is_discrete: Whether the system is discrete-time
        L: Estimator gain matrix (computed)
        P: Steady-state error covariance (computed)
    """

    def __init__(
        self,
        A: np.ndarray = None,
        C: np.ndarray = None,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        sys: ct.StateSpace = None,
        validator: IEstimatorValidator = None,
    ):
        """
        Initialize the Linear Quadratic Estimator.

        Can be initialized either with explicit matrices (A, C, Q, R) or with
        a control.StateSpace system object.

        Args:
            A: State transition matrix (n_states x n_states). Required if sys is None.
            C: Observation matrix (n_measurements x n_states). Required if sys is None.
            Q: Process noise covariance (n_states x n_states). Required if sys is None.
            R: Measurement noise covariance (n_measurements x n_measurements). Required if sys is None.
            sys: Control system StateSpace object. If provided, A, B, and C are extracted from it.
            validator: Validation strategy (default: LinearEstimatorValidator)

        Raises:
            ValueError: If matrix dimensions are invalid or matrices have wrong properties
            ValueError: If neither (A, C, Q, R) nor sys is provided
        """
        # Use default validator if none provided
        if validator is None:
            validator = LinearEstimatorValidator()

        self.validator = validator

        # Extract system matrices
        if sys is not None:
            # Extract from StateSpace system
            if not isinstance(sys, ct.StateSpace):
                raise ValueError("sys must be a control.StateSpace object")

            self.A = np.array(sys.A)
            self.B = np.array(sys.B)
            self.C = np.array(sys.C)
            self.is_discrete = sys.dt is not None and sys.dt > 0

            # Still need Q and R to be provided
            if Q is None or R is None:
                raise ValueError(
                    "Q and R matrices must be provided even when using sys"
                )
            self.Q = Q
            self.R = R
        else:
            # All matrices must be provided
            if A is None or C is None or Q is None or R is None:
                raise ValueError("Either sys or all of (A, C, Q, R) must be provided")

            self.A = A
            self.C = C
            self.Q = Q
            self.R = R
            self.is_discrete = False  # Default to continuous unless specified
            # Create dummy B matrix if not provided via sys
            n_states = self.A.shape[0]
            self.B = np.zeros((n_states, 1))

        # Validate matrices using LinearEstimatorValidator
        n_states = self.A.shape[0]
        x0_dummy = np.zeros(n_states)
        P_dummy = np.eye(n_states)

        # Determine dt for validation
        dt_for_validation = (1.0, 1.0) if self.is_discrete else None

        validation_result = self.validator.validate_initialization(
            A=self.A,
            B=self.B,
            C=self.C,
            Q=self.Q,
            R=self.R,
            P=P_dummy,
            x0=x0_dummy,
            dt=dt_for_validation,
        )

        # Check for errors (warnings are okay)
        if validation_result.has_errors():
            error_messages = "\n".join(validation_result.get_errors())
            raise ValueError(
                f"Linear Quadratic Estimator initialization failed:\n{error_messages}"
            )

        # Store computed results
        self._L = None
        self._P = None
        self._E = None

    def compute_estimator_gain(self) -> tuple:
        """
        Compute the optimal LQE gain matrix L and steady-state covariance P.

        Solves the dual algebraic Riccati equation to find the optimal estimator
        gain matrix L such that the steady-state estimation error is minimized.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (L, P)
                L: Estimator gain matrix (n_states x n_measurements)
                P: Steady-state error covariance (n_states x n_states)

        Raises:
            ValueError: If the LQE problem cannot be solved
            ValueError: If the solution results in an unstable estimator
        """
        if self._L is not None and self._P is not None:
            return self._L, self._P

        try:
            if self.is_discrete:
                # Discrete-time LQE: dlqe(A, G, C, QN, RN)
                # where G is the process noise input matrix (typically identity)
                # Q is process noise covariance, R is measurement noise covariance
                G = np.eye(self.A.shape[0])  # Assume process noise affects all states
                self._L, self._P, self._E = ct.dlqe(self.A, G, self.C, self.Q, self.R)
            else:
                # Continuous-time LQE: lqe(A, G, C, QN, RN)
                # where G is the process noise input matrix (typically identity)
                # Q is process noise covariance, R is measurement noise covariance
                G = np.eye(self.A.shape[0])  # Assume process noise affects all states
                self._L, self._P, self._E = ct.lqe(self.A, G, self.C, self.Q, self.R)
        except Exception as e:
            raise ValueError(f"Failed to solve LQE problem: {e}")

        # Check stability of the estimator using eigenvalues from control library
        if self.is_discrete:
            # Discrete-time: eigenvalues must be inside unit circle
            max_magnitude = np.max(np.abs(self._E))
            if max_magnitude >= 1.0:
                raise ValueError(
                    f"LQE solution results in unstable estimator "
                    f"(max eigenvalue magnitude: {max_magnitude:.6f})"
                )
        else:
            # Continuous-time: eigenvalues must have negative real parts
            max_real_part = np.max(np.real(self._E))
            if max_real_part >= 0:
                raise ValueError(
                    f"LQE solution results in unstable estimator "
                    f"(max eigenvalue real part: {max_real_part:.6f})"
                )

        return self._L, self._P

    def get_L(self) -> np.ndarray:
        """
        Get the computed estimator gain matrix L.

        Returns:
            Estimator gain matrix L if already computed, otherwise computes it first
        """
        if self._L is None:
            self.compute_estimator_gain()
        return self._L

    def get_P(self) -> np.ndarray:
        """
        Get the computed steady-state error covariance matrix P.

        Returns:
            Steady-state error covariance P if already computed, otherwise computes it first
        """
        if self._P is None:
            self.compute_estimator_gain()
        return self._P

    def get_A(self) -> np.ndarray:
        """
        Get the A matrix for the system.

        Returns:
            A matrix for the system
        """
        return self.A

    def get_C(self) -> np.ndarray:
        """
        Get the C matrix for the system.

        Returns:
            C matrix for the system
        """
        return self.C

    def is_discrete_time(self) -> bool:
        """
        Check if the system is discrete-time.

        Returns:
            True if discrete-time, False if continuous-time
        """
        return self.is_discrete

    def set_discrete_time(self, is_discrete: bool):
        """
        Set whether the system is discrete-time or continuous-time.

        This should be called before compute_estimator_gain() if you need to
        override the default (continuous) or the value inferred from sys.

        Args:
            is_discrete: True for discrete-time, False for continuous-time
        """
        if self._L is not None:
            raise ValueError(
                "Cannot change discrete/continuous mode after gain has been computed"
            )
        self.is_discrete = is_discrete
