"""
Discrete-time Kalman Filter implementation.

This module provides a standard discrete-time Kalman filter for state estimation
in linear time-invariant systems with discrete measurements.
"""

import numpy as np
import warnings
from .estimator_abstractions import AbstractEstimatorHandler
from .validator_abstractions import IEstimatorValidator
from .linear_estimator_validator import LinearEstimatorValidator


class DiscreteKalman(AbstractEstimatorHandler):
    """
    Standard Discrete-Time Kalman Filter for state estimation.

    This class implements the classical discrete-time Kalman Filter where both
    system dynamics and measurements are discrete. The filter performs prediction
    and update steps at each time step.

    The filter operates in two main steps:
        1. Prediction: Propagate state and covariance forward using system model
        2. Update: Refine estimates using new measurement (Joseph form for stability)

    Key Features:
    - Discrete-time state propagation
    - Joseph form covariance update for numerical stability
    - Uses Strategy Pattern for validation (delegated to validator)
    - Every call to estimate_states performs both prediction and update

    Attributes:
        A: Discrete-time state transition matrix
        B: Discrete-time control input matrix
        C: Observation matrix (output matrix, following Stanford notation)
        Q: Discrete-time process noise covariance
        R: Measurement noise covariance
        P: Estimate error covariance
        x_est: Current state estimate
        validator: Validation strategy for initialization and runtime checks
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        P: np.ndarray,
        x0: np.ndarray,
        validator: IEstimatorValidator = None,
    ):
        """
        Initialize the Discrete Kalman Filter with system matrices and initial conditions.

        Args:
            A: Discrete-time state transition matrix (n_states x n_states)
            B: Discrete-time control input matrix (n_states x n_inputs)
            C: Observation/output matrix (n_measurements x n_states)
            Q: Discrete-time process noise covariance (n_states x n_states)
            R: Measurement noise covariance (n_measurements x n_measurements)
            P: Initial estimate error covariance (n_states x n_states)
            x0: Initial state estimate (n_states,)
            validator: Validation strategy (default: LinearEstimatorValidator)

        Raises:
            ValueError: If validation fails with errors

        Note:
            A, B, and Q should be discrete-time matrices. If you have continuous-time
            matrices, use continuum_robot.utils.discretization utilities to convert them.
        """
        # Use default validator if none provided
        if validator is None:
            validator = LinearEstimatorValidator()

        self.validator = validator

        # Validate initialization parameters
        # Note: We pass dt=1.0 as a placeholder since DiscreteKalman doesn't use dt
        validation_result = self.validator.validate_initialization(
            A, B, C, Q, R, P, x0, dt=1.0
        )

        # Raise errors if validation failed
        if validation_result.has_errors():
            error_messages = "\n".join(validation_result.get_errors())
            raise ValueError(
                f"Discrete Kalman filter initialization failed:\n{error_messages}"
            )

        # Issue warnings for non-critical issues
        for warning_msg in validation_result.get_warnings():
            warnings.warn(warning_msg)

        # Store system matrices (make copies to prevent external modification)
        self.A = A.copy()
        self.B = B.copy()
        self.C = C.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.P = P.copy()
        self.x_est = x0.copy()

        # Store dimensions for runtime validation
        self.n_states = A.shape[0]
        self.n_measurements = C.shape[0]
        self.n_inputs = B.shape[1] if B.ndim > 1 else 1

    def _predict(self, u: np.ndarray) -> tuple:
        """
        Predict the next state and estimate error covariance.

        Implements the discrete-time prediction equations:
            x_pred = A @ x + B @ u
            P_pred = A @ P @ A.T + Q

        Args:
            u: Current input vector (n_inputs,)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (x_pred, P_pred)
                x_pred: Predicted state vector (n_states,)
                P_pred: Predicted covariance matrix (n_states, n_states)
        """
        x_pred = self.A @ self.x_est + self.B @ u
        P_pred = self.A @ self.P @ self.A.T + self.Q
        return x_pred, P_pred

    def _update(self, y: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray) -> None:
        """
        Update the state estimate and covariance using the new measurement.

        Implements the discrete-time update equations using Joseph form for
        numerical stability:
            K = P_pred @ C.T @ inv(C @ P_pred @ C.T + R)
            x_est = x_pred + K @ (y - C @ x_pred)
            P = (I - K @ C) @ P_pred @ (I - K @ C).T + K @ R @ K.T  (Joseph form)

        Args:
            y: Current measurement vector (n_measurements,)
            x_pred: Predicted state vector (n_states,)
            P_pred: Predicted estimate error covariance (n_states, n_states)
        """
        # Compute innovation
        y_pred = self.C @ x_pred
        innovation = y - y_pred

        # Compute innovation covariance
        S = self.C @ P_pred @ self.C.T + self.R

        # Check for singular innovation covariance
        if np.linalg.det(S) < 1e-12:
            warnings.warn(
                "Innovation covariance is nearly singular, using pseudoinverse"
            )
            S_inv = np.linalg.pinv(S)
        else:
            S_inv = np.linalg.inv(S)

        # Compute Kalman gain
        K = P_pred @ self.C.T @ S_inv

        # Update state estimate
        self.x_est = x_pred + K @ innovation

        # Joseph form covariance update for numerical stability
        # P = (I - K*C)*P_pred*(I - K*C)' + K*R*K'
        I_KC = np.eye(self.n_states) - K @ self.C
        self.P = I_KC @ P_pred @ I_KC.T + K @ self.R @ K.T

        # Ensure P remains positive semidefinite (symmetrize and regularize if needed)
        self.P = 0.5 * (self.P + self.P.T)  # Ensure symmetry

        if not self._is_positive_semidefinite(self.P):
            warnings.warn(
                "Estimate covariance became non-positive semidefinite, applying regularization"
            )
            eigenvals, eigenvecs = np.linalg.eigh(self.P)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Regularize
            self.P = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def estimate_states(self, y: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """
        Estimate current states given observations and input.

        Performs both prediction and update steps:
        1. Prediction: Propagate state forward using system model
        2. Update: Incorporate measurement to refine estimate

        Args:
            y: Current measurement vector (n_measurements,)
            u: Current input vector (n_inputs,)
            t: Current time (unused for discrete filter, kept for interface consistency)

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

        # Perform prediction step
        x_pred, P_pred = self._predict(u)

        # Perform measurement update
        self._update(y, x_pred, P_pred)

        # Return updated state (make copy to prevent external modification)
        return self.x_est.copy()

    @staticmethod
    def _is_positive_semidefinite(matrix: np.ndarray) -> bool:
        """Check if matrix is positive semidefinite using eigenvalues."""
        try:
            eigenvals = np.linalg.eigvals(matrix)
            return np.all(eigenvals >= -1e-8)  # Small tolerance for numerical errors
        except Exception:
            return False
