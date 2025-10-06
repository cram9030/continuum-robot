import numpy as np
import warnings
from .estimator_abstractions import AbstractEstimatorHandler
from .validator_abstractions import IEstimatorValidator
from .linear_estimator_validator import LinearEstimatorValidator


class HybridContinuousDiscreteKalman(AbstractEstimatorHandler):
    """
    Hybrid Continuous-Discrete Kalman Filter for state estimation in continuum robots.

    This class implements a hybrid continuous-discrete Kalman Filter where the system
    dynamics are continuous but measurements arrive at discrete time intervals. The filter
    performs prediction continuously and updates only when new measurements are available
    based on the time elapsed since the last measurement.

    The filter operates in two main steps:
        1. Prediction: Continuously estimate state evolution based on system dynamics
        2. Update: Discretely refine estimates when new measurements arrive (dt elapsed)

    Key Features:
    - Time-aware measurement handling: only updates when dt time has elapsed
    - Returns predicted states for intermediate time queries
    - Thread-safe for real-time robotics applications
    - Uses Strategy Pattern for validation (delegated to validator)

    Attributes:
        A: State transition matrix (continuous-time)
        B: Control input matrix
        C: Observation matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        P: Estimate error covariance
        x_est: Current state estimate
        dt: Discrete measurement interval
        last_update_time: Time of last measurement update
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
        dt: float = 0.01,
        validator: IEstimatorValidator = None,
    ):
        """
        Initialize the Kalman Filter with system matrices and initial conditions.

        Args:
            A: State transition matrix (n_states x n_states)
            B: Control input matrix (n_states x n_inputs)
            C: Observation matrix (n_measurements x n_states)
            Q: Process noise covariance (n_states x n_states)
            R: Measurement noise covariance (n_measurements x n_measurements)
            P: Initial estimate error covariance (n_states x n_states)
            x0: Initial state estimate (n_states,)
            dt: Discrete time step for integration (must be positive)
            validator: Validation strategy (default: LinearEstimatorValidator)

        Raises:
            ValueError: If validation fails with errors
        """
        # Use default validator if none provided
        if validator is None:
            validator = LinearEstimatorValidator()

        self.validator = validator

        # Validate initialization parameters
        validation_result = self.validator.validate_initialization(
            A, B, C, Q, R, P, x0, dt
        )

        # Raise errors if validation failed
        if validation_result.has_errors():
            error_messages = "\n".join(validation_result.get_errors())
            raise ValueError(f"Kalman filter initialization failed:\n{error_messages}")

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
        self.dt = float(dt)
        self.last_update_time = 0.0  # Initialize to 0 for first call
        self._first_call = True  # Track if this is the first estimate call

        # Store dimensions for runtime validation
        self.n_states = A.shape[0]
        self.n_measurements = C.shape[0]
        self.n_inputs = B.shape[1] if B.ndim > 1 else 1

    @staticmethod
    def _is_positive_semidefinite(matrix):
        """Check if matrix is positive semidefinite."""
        eigenvals = np.linalg.eigvals(matrix)
        return np.all(eigenvals >= -1e-8)  # Small tolerance for numerical errors

    def _predict(self, u: np.ndarray) -> tuple:
        """
        Predict the next state and estimate error covariance.

        Args:
            u: Current input vector

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted state derivative and covariance derivative
        """
        xdot_pred = self.A @ self.x_est + self.B @ u
        Pdot_pred = self.A @ self.P + self.P @ self.A.T + self.Q
        return xdot_pred, Pdot_pred

    def _update(self, y: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray) -> None:
        """
        Update the state estimate and estimate error covariance using the new measurement.

        Uses Joseph form for numerical stability in covariance update.

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

        # Compute Kalman gain
        K = P_pred @ self.C.T @ np.linalg.inv(S)

        # Update state estimate
        self.x_est = x_pred + K @ innovation

        # Joseph form covariance update for numerical stability
        I_KC = np.eye(self.n_states) - K @ self.C
        self.P = I_KC @ P_pred

        # Ensure P remains positive semidefinite
        if not self._is_positive_semidefinite(self.P):
            warnings.warn(
                "Estimate covariance became non-positive semidefinite, applying regularization"
            )
            eigenvals, eigenvecs = np.linalg.eigh(self.P)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Regularize
            self.P = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def estimate_states(self, y: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """
        Estimate current states given observations, input, and time.

        This method implements hybrid continuous-discrete estimation using matrix exponentials
        for accurate continuous-time propagation:
        - If t - last_update_time < dt: performs prediction only and returns predicted state
        - If t - last_update_time >= dt: performs prediction and measurement update, updates last_update_time

        The prediction uses:
            x_pred = expm(A * dt) @ x + integral(expm(A * tau) @ B @ u, 0, dt)
            P_pred = expm(A * dt) @ P @ expm(A * dt).T + Q_discrete

        Args:
            y: Current measurement vector (n_measurements,) - used only when dt has elapsed
            u: Current input vector (n_inputs,)
            t: Current time (must be non-decreasing between calls)

        Returns:
            np.ndarray: Current state vector estimation [positions, velocities] (n_states,)
                       - Predicted state if measurement update is not due
                       - Updated state if measurement update is performed

        Raises:
            ValueError: If runtime validation fails
        """
        from scipy.linalg import expm

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

        # Validate time type
        if not isinstance(t, (int, float)):
            raise TypeError(f"Time t must be numeric, got {type(t)}")

        # Calculate time since last update
        time_since_update = t - self.last_update_time

        # Compute discrete-time state transition matrix using matrix exponential
        A_discrete = expm(self.A * time_since_update)

        # Compute discrete-time control input matrix
        # B_discrete = integral(expm(A * tau) @ B, 0, dt)
        # For small dt, approximation: B_discrete ≈ B * dt
        # For better accuracy, use: inv(A) @ (expm(A*dt) - I) @ B
        if np.linalg.matrix_rank(self.A) == self.n_states:
            # A is full rank, use exact formula
            B_discrete = np.linalg.solve(
                self.A, (A_discrete - np.eye(self.n_states)) @ self.B
            )
        else:
            # A is singular or near-singular, use first-order approximation
            B_discrete = self.B * time_since_update

        # Perform prediction step using discrete-time transition
        x_pred = A_discrete @ self.x_est + B_discrete @ u

        # Propagate covariance
        # P_pred = A_discrete @ P @ A_discrete.T + Q_discrete
        # For continuous-time Q, discretize as Q_discrete ≈ Q * dt
        Q_discrete = self.Q * time_since_update
        P_pred = A_discrete @ self.P @ A_discrete.T + Q_discrete

        # Decide whether to perform measurement update
        # Update if: (1) first call, or (2) dt time has elapsed since last update
        if self._first_call or time_since_update >= self.dt:
            # Time for measurement update
            self._update(y, x_pred, P_pred)
            self.last_update_time = t  # Update to current time
            self._first_call = False  # No longer the first call
            return self.x_est.copy()  # Return updated state
        else:
            # Return predicted state without update
            return x_pred.copy()  # Return predicted state only
