import numpy as np
import warnings
from .estimator_abstractions import AbstractEstimatorHandler


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
    - Comprehensive parameter validation with robotics-specific warnings

    Attributes:
        A: State transition matrix (continuous-time)
        B: Control input matrix
        H: Observation matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        P: Estimate error covariance
        x_est: Current state estimate
        dt: Discrete measurement interval
        last_update_time: Time of last measurement update
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        P: np.ndarray,
        x0: np.ndarray,
        dt: float = 0.01,
    ):
        """
        Initialize the Kalman Filter with system matrices and initial conditions.

        Args:
            A: State transition matrix (n_states x n_states)
            B: Control input matrix (n_states x n_inputs)
            H: Observation matrix (n_measurements x n_states)
            Q: Process noise covariance (n_states x n_states)
            R: Measurement noise covariance (n_measurements x n_measurements)
            P: Initial estimate error covariance (n_states x n_states)
            x0: Initial state estimate (n_states,)
            dt: Discrete time step for integration (must be positive)

        Raises:
            ValueError: If matrices have incompatible dimensions or invalid properties
            TypeError: If inputs are not numpy arrays or dt is not numeric
        """
        # Input validation
        self._validate_initialization_inputs(A, B, H, Q, R, P, x0, dt)

        self.A = A.copy()
        self.B = B.copy()
        self.H = H.copy()
        self.Q = Q.copy()
        self.R = R.copy()
        self.P = P.copy()
        self.x_est = x0.copy()
        self.dt = float(dt)
        self.last_update_time = 0.0  # Initialize to 0 for first call
        self._first_call = True  # Track if this is the first estimate call

        # Robot-specific validation warnings
        self._check_estimation_specific_properties()

    def _validate_initialization_inputs(self, A, B, H, Q, R, P, x0, dt):
        """Validate initialization inputs for robotics applications."""
        matrices = {"A": A, "B": B, "H": H, "Q": Q, "R": R, "P": P, "x0": x0}

        # Type validation
        self._validate_types(matrices, dt)

        # Dimension validation
        n_states = A.shape[0]
        n_measurements = H.shape[0]
        self._validate_dimensions(A, B, H, Q, R, P, x0, n_states, n_measurements)

        # Value validation
        self._validate_values(matrices)

        # Matrix properties validation
        self._validate_covariance_matrices(Q, R, P)

    def _validate_types(self, matrices, dt):
        """Validate types of input matrices and dt."""
        for name, matrix in matrices.items():
            if not isinstance(matrix, np.ndarray):
                raise TypeError(
                    f"Matrix {name} must be a numpy array, got {type(matrix)}"
                )
            if not np.issubdtype(matrix.dtype, np.floating):
                warnings.warn(
                    f"Matrix {name} should be floating point type for numerical stability, got {matrix.dtype}"
                )

        if not isinstance(dt, (int, float)) or dt <= 0:
            raise ValueError(f"dt must be a positive number, got {dt}")

    def _validate_dimensions(self, A, B, H, Q, R, P, x0, n_states, n_measurements):
        """Validate dimensions of system matrices."""
        if A.shape != (n_states, n_states):
            raise ValueError(f"A must be square ({n_states}x{n_states}), got {A.shape}")
        if B.shape[0] != n_states:
            raise ValueError(f"B must have {n_states} rows to match A, got {B.shape}")
        if H.shape[1] != n_states:
            raise ValueError(
                f"H must have {n_states} columns to match A, got {H.shape}"
            )
        if Q.shape != (n_states, n_states):
            raise ValueError(
                f"Q must match A dimensions ({n_states}x{n_states}), got {Q.shape}"
            )
        if R.shape != (n_measurements, n_measurements):
            raise ValueError(
                f"R must be {n_measurements}x{n_measurements}, got {R.shape}"
            )
        if P.shape != (n_states, n_states):
            raise ValueError(
                f"P must match A dimensions ({n_states}x{n_states}), got {P.shape}"
            )
        if x0.shape != (n_states,):
            raise ValueError(f"x0 must have {n_states} elements, got {x0.shape}")

        # Check for empty matrices
        if A.size == 0 or B.size == 0 or H.size == 0:
            raise ValueError("System matrices cannot be empty")

    def _validate_values(self, matrices):
        """Validate that matrices don't contain NaN or infinite values."""
        for name, matrix in matrices.items():
            if np.any(np.isnan(matrix)):
                raise ValueError(f"Matrix {name} contains NaN values")
            if np.any(np.isinf(matrix)):
                raise ValueError(f"Matrix {name} contains infinite values")

    def _validate_covariance_matrices(self, Q, R, P):
        """Validate covariance matrix properties."""
        if not self._is_positive_semidefinite(Q):
            raise ValueError("Process noise covariance Q must be positive semidefinite")
        if not self._is_positive_definite(R):
            raise ValueError("Measurement noise covariance R must be positive definite")
        if not self._is_positive_semidefinite(P):
            raise ValueError("Initial error covariance P must be positive semidefinite")

    def _check_estimation_specific_properties(self):
        """Check for estimation-specific properties and issue warnings."""
        n_states = self.A.shape[0]

        # Check if state dimension suggests position/velocity structure
        if n_states % 2 == 0:
            n_dof = n_states // 2
            if n_dof > 50:
                warnings.warn(
                    f"Large number of DOFs ({n_dof}) detected. Consider model reduction for real-time applications."
                )

        # Check eigenvalues for stability
        eigenvals = np.linalg.eigvals(self.A)
        max_real_part = np.max(np.real(eigenvals))
        if max_real_part > 0:
            warnings.warn(
                f"System matrix A may be unstable (max eigenvalue real part: {max_real_part:.6f})"
            )

        # Check condition numbers for numerical issues
        cond_A = np.linalg.cond(self.A)
        if cond_A > 1e12:
            warnings.warn(
                f"System matrix A is ill-conditioned (condition number: {cond_A:.2e})"
            )

        cond_P = np.linalg.cond(self.P)
        if cond_P > 1e12:
            warnings.warn(
                f"Initial covariance P is ill-conditioned (condition number: {cond_P:.2e})"
            )

        # Check for extremely large time steps (robotics safety)
        if self.dt > 1.0:
            warnings.warn(
                f"Large time step dt={self.dt}s may cause numerical instability"
            )

    @staticmethod
    def _is_positive_definite(matrix):
        """Check if matrix is positive definite."""
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def _is_positive_semidefinite(matrix):
        """Check if matrix is positive semidefinite."""
        eigenvals = np.linalg.eigvals(matrix)
        return np.all(eigenvals >= -1e-8)  # Small tolerance for numerical errors

    def _predict(self, u: np.ndarray) -> np.ndarray:
        """
        Predict the next state and estimate error covariance.

        Args:
            u: Current input vector

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted state derivative and covariance derivative

        Raises:
            ValueError: If input dimensions don't match system dimensions
            TypeError: If input is not a numpy array
        """
        # Validate input
        if not isinstance(u, np.ndarray):
            raise TypeError(f"Input u must be numpy array, got {type(u)}")

        expected_input_dim = self.B.shape[1] if self.B.ndim > 1 else 1
        if u.shape != (expected_input_dim,):
            raise ValueError(
                f"Input u must have shape ({expected_input_dim},), got {u.shape}"
            )

        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            raise ValueError("Input u contains NaN or infinite values")

        try:
            xdot_pred = self.A @ self.x_est + self.B @ u
            Pdot_pred = self.A @ self.P + self.P @ self.A.T + self.Q
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Linear algebra error during prediction: {e}")

        return xdot_pred, Pdot_pred

    def _update(self, y: np.ndarray, x_pred: np.ndarray, P_pred: np.ndarray) -> None:
        """
        Update the state estimate and estimate error covariance using the new measurement.

        Args:
            y: Current measurement vector (n_measurements,)
            x_pred: Predicted state vector (n_states,)
            P_pred: Predicted estimate error covariance (n_states, n_states)

        Raises:
            ValueError: If input dimensions don't match or matrices are singular
            TypeError: If inputs are not numpy arrays
        """
        # Input validation
        for name, array in [("y", y), ("x_pred", x_pred), ("P_pred", P_pred)]:
            if not isinstance(array, np.ndarray):
                raise TypeError(f"{name} must be numpy array, got {type(array)}")
            if np.any(np.isnan(array)) or np.any(np.isinf(array)):
                raise ValueError(f"{name} contains NaN or infinite values")

        # Dimension validation
        n_measurements = self.H.shape[0]
        n_states = self.A.shape[0]

        if y.shape != (n_measurements,):
            raise ValueError(
                f"Measurement y must have shape ({n_measurements},), got {y.shape}"
            )
        if x_pred.shape != (n_states,):
            raise ValueError(
                f"Predicted state x_pred must have shape ({n_states},), got {x_pred.shape}"
            )
        if P_pred.shape != (n_states, n_states):
            raise ValueError(
                f"Predicted covariance P_pred must have shape ({n_states}, {n_states}), got {P_pred.shape}"
            )

        try:
            y_pred = self.H @ x_pred
            innovation = y - y_pred
            S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance

            # Check for singular innovation covariance
            if np.linalg.det(S) < 1e-12:
                warnings.warn(
                    "Innovation covariance is nearly singular, using pseudoinverse"
                )

            K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain

            self.x_est = x_pred + K @ innovation

            # Joseph covariance update form for numerical stability
            I_KH = np.eye(n_states) - K @ self.H
            self.P = I_KH @ P_pred

            # Ensure P remains positive semidefinite
            if not self._is_positive_semidefinite(self.P):
                warnings.warn(
                    "Estimate covariance became non-positive semidefinite, applying regularization"
                )
                eigenvals, eigenvecs = np.linalg.eigh(self.P)
                eigenvals = np.maximum(eigenvals, 1e-12)  # Regularize
                self.P = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        except np.linalg.LinAlgError as e:
            raise ValueError(f"Linear algebra error during update: {e}")

    def estimate_states(self, y: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """
        Estimate current states given observations, input, and time.

        This method implements hybrid continuous-discrete estimation:
        - If t - last_update_time < dt: performs prediction only and returns predicted state
        - If t - last_update_time >= dt: performs prediction and measurement update, updates last_update_time

        Args:
            y: Current measurement vector (n_measurements,) - used only when dt has elapsed
            u: Current input vector (n_inputs,)
            t: Current time (must be non-decreasing between calls)

        Returns:
            np.ndarray: Current state vector estimation [positions, velocities] (n_states,)
                       - Predicted state if measurement update is not due
                       - Updated state if measurement update is performed

        Raises:
            ValueError: If inputs have wrong dimensions, contain invalid values, or t decreases
            TypeError: If inputs are not numpy arrays
        """
        # Input validation
        if not isinstance(y, np.ndarray):
            raise TypeError(f"Measurement y must be numpy array, got {type(y)}")
        if not isinstance(u, np.ndarray):
            raise TypeError(f"Input u must be numpy array, got {type(u)}")
        if not isinstance(t, (int, float)):
            raise TypeError(f"Time t must be numeric, got {type(t)}")

        # Validate measurement dimensions (even if not used, for consistency)
        n_measurements = self.H.shape[0]
        if y.shape != (n_measurements,):
            raise ValueError(
                f"Measurement y must have shape ({n_measurements},), got {y.shape}"
            )

        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("Measurement y contains NaN or infinite values")

        try:
            # Always perform prediction step
            xdot_pred, Pdot_pred = self._predict(u)

            # Calculate time since last update
            time_since_update = t - self.last_update_time

            # Perform prediction step
            x_pred = self.x_est + xdot_pred * time_since_update
            P_pred = self.P + Pdot_pred * time_since_update

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

        except (ValueError, TypeError) as e:
            # Re-raise validation errors
            raise e
        except Exception as e:
            # Catch any other unexpected errors
            raise ValueError(f"Unexpected error during state estimation: {e}")
