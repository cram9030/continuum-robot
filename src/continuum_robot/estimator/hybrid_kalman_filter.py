import numpy as np
import warnings
from scipy.integrate import solve_ivp
from .estimator_abstractions import AbstractEstimatorHandler
from .validator_abstractions import IEstimatorValidator
from .linear_estimator_validator import LinearEstimatorValidator


class HybridContinuousDiscreteKalman(AbstractEstimatorHandler):
    """
    Hybrid Continuous-Discrete Kalman Filter for state estimation in continuum robots.

    This class implements a hybrid continuous-discrete Kalman Filter where the system
    dynamics are continuous but measurements arrive at discrete time intervals. The filter
    performs prediction continuously using solve_ivp integration and updates only when
    new measurements are available based on the time elapsed since the last measurement.

    The filter operates in two main steps:
        1. Prediction: Continuously integrate state evolution x̂_{i|i-1} and covariance P_{i|i-1}
        2. Update: Discretely refine estimates to x̂_{k|k} and P_{k|k} when measurements arrive

    Notation:
        x̂_{i|i-1}: Predicted state at time i given information up to previous predict step
        P_{i|i-1}: Predicted error covariance at time i given information up to previous predict step
        x̂_{k|k}: Updated state at measurement time k given measurement at time k
        P_{k|k}: Updated error covariance at measurement time k given measurement at time k

    Key Features:
    - Time-aware measurement handling: only updates when dt time has elapsed
    - Returns predicted states for intermediate time queries
    - Uses solve_ivp for accurate continuous-time integration
    - Thread-safe for real-time robotics applications
    - Uses Strategy Pattern for validation (delegated to validator)

    Attributes:
        A: State transition matrix (continuous-time)
        B: Control input matrix
        C: Observation matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        P_est: Estimate error covariance P_{k|k} after measurement update
        x_est: State estimate x̂_{k|k} after measurement update
        P_pred: Predicted error covariance P_{i|i-1}
        x_pred: Predicted state x̂_{i|i-1}
        dt: Discrete measurement interval
        last_update_time: Time of last measurement update
        last_pred_time: Time of last prediction
        integrator_options: Dictionary of solve_ivp solver options
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
        integrator_options: dict | None = None,
    ):
        """
        Initialize the Kalman Filter with system matrices and initial conditions.

        Args:
            A: State transition matrix (n_states x n_states)
            B: Control input matrix (n_states x n_inputs)
            C: Observation matrix (n_measurements x n_states)
            Q: Process noise covariance (n_states x n_states)
            R: Measurement noise covariance (n_measurements x n_measurements)
            P: Initial estimate error covariance P_{0|0} (n_states x n_states)
            x0: Initial state estimate x̂_{0|0} (n_states,)
            dt: Discrete time step for measurement updates (must be positive)
            validator: Validation strategy (default: LinearEstimatorValidator)
            integrator_options: Optional dictionary of solver options for solve_ivp
                               (e.g., {'method': 'RK45', 'rtol': 1e-6, 'atol': 1e-9})
                               If None, defaults to RK45 with standard tolerances

        Raises:
            ValueError: If validation fails with errors
        """
        # Use default validator if none provided
        if validator is None:
            validator = LinearEstimatorValidator()

        self.validator = validator

        # Validate initialization parameters
        # For hybrid continuous-discrete: state equation is continuous (None),
        # measurement is discrete (dt)
        validation_result = self.validator.validate_initialization(
            A, B, C, Q, R, P, x0, dt=(None, dt)
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
        self.P_est = P.copy()  # P_{k|k} - estimate error covariance after update
        self.x_est = x0.copy()  # x̂_{k|k} - state estimate after update
        self.dt = float(dt)
        self.last_update_time = 0.0  # Initialize to 0 for first call
        self._first_call = True  # Track if this is the first estimate call

        # Initialize prediction states
        self.x_pred = x0.copy()  # x̂_{i|i-1} - predicted state
        self.P_pred = P.copy()  # P_{i|i-1} - predicted error covariance
        self.last_pred_time = 0.0  # Time of last prediction

        # Set integrator options with defaults
        if integrator_options is None:
            self.integrator_options = {"method": "RK45", "rtol": 1e-6, "atol": 1e-9}
        else:
            # Ensure method is specified, default to RK45 if not
            self.integrator_options = integrator_options.copy()
            if "method" not in self.integrator_options:
                self.integrator_options["method"] = "RK45"
            if "rtol" not in self.integrator_options:
                self.integrator_options["rtol"] = 1e-6
            if "atol" not in self.integrator_options:
                self.integrator_options["atol"] = 1e-9

        # Store dimensions for runtime validation
        self.n_states = A.shape[0]
        self.n_measurements = C.shape[0]
        self.n_inputs = B.shape[1] if B.ndim > 1 else 1

    def _predict(self, u: np.ndarray, t: float) -> None:
        """
        Predict the state and estimate error covariance to time t using solve_ivp.

        This method integrates the continuous-time dynamics from the most recent time
        (either last_pred_time or last_update_time) to the current time t.
        Updates self.x_pred and self.P_pred.

        Dynamics:
            dx̂/dt = A @ x̂ + B @ u
            dP/dt = A @ P + P @ A^T + Q

        Args:
            u: Current input vector (assumed constant over integration interval)
            t: Current time to predict to
        """
        # Determine which state and covariance to use as initial conditions
        # Use most recent of prediction or update
        if self.last_pred_time >= self.last_update_time:
            x_init = self.x_pred
            P_init = self.P_pred
            t_start = self.last_pred_time
        else:
            x_init = self.x_est
            P_init = self.P_est
            t_start = self.last_update_time

        # If we're already at the current time, no need to integrate
        if abs(t - t_start) < 1e-10:
            return

        # Create combined state vector [x, P.flatten()]
        n = self.n_states
        combined_init = np.concatenate([x_init, P_init.flatten()])

        # Define the combined dynamics function
        def combined_dynamics(t_val, combined_state):
            x = combined_state[:n]
            P_flat = combined_state[n:]
            P = P_flat.reshape((n, n))

            # State dynamics: dx/dt = A @ x + B @ u
            xdot = self.A @ x + self.B @ u

            # Covariance dynamics: dP/dt = A @ P + P @ A^T + Q
            Pdot = self.A @ P + P @ self.A.T + self.Q
            Pdot_flat = Pdot.flatten()

            return np.concatenate([xdot, Pdot_flat])

        # Integrate using solve_ivp
        sol = solve_ivp(
            combined_dynamics, [t_start, t], combined_init, **self.integrator_options
        )

        # Extract final state and covariance
        combined_final = sol.y[:, -1]
        self.x_pred = combined_final[:n]
        self.P_pred = combined_final[n:].reshape((n, n))
        self.last_pred_time = t

    def _update(self, y: np.ndarray) -> None:
        """
        Update the state estimate and estimate error covariance using the new measurement.

        This performs the discrete measurement update step, converting predicted estimates
        x̂_{k|i-1} and P_{k|i-1} to updated estimates x̂_{k|k} and P_{k|k}.

        Uses Joseph form for numerical stability in covariance update.

        Args:
            y: Current measurement vector (n_measurements,)
        """
        # Compute innovation using predicted state
        y_pred = self.C @ self.x_pred
        innovation = y - y_pred

        # Compute innovation covariance
        S = self.C @ self.P_pred @ self.C.T + self.R

        # Check for singular innovation covariance
        if np.linalg.det(S) < 1e-12:
            warnings.warn(
                "Innovation covariance is nearly singular, using pseudoinverse"
            )

        # Compute Kalman gain
        K = self.P_pred @ self.C.T @ np.linalg.inv(S)

        # Update state estimate: x̂_{k|k} = x̂_{k|i-1} + K @ (y - ŷ)
        self.x_est = self.x_pred + K @ innovation

        # Joseph form covariance update for numerical stability
        # P_{k|k} = (I - K @ C) @ P_{k|i-1}
        I_KC = np.eye(self.n_states) - K @ self.C
        self.P_est = I_KC @ self.P_pred

        # Ensure P_est remains positive semidefinite
        if not LinearEstimatorValidator._is_positive_semidefinite(self.P_est):
            warnings.warn(
                "Estimate covariance became non-positive semidefinite, applying regularization"
            )
            eigenvals, eigenvecs = np.linalg.eigh(self.P_est)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Regularize
            self.P_est = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def estimate_states(self, y: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """
        Estimate current states given observations, input, and time.

        This method implements hybrid continuous-discrete estimation using solve_ivp
        for continuous-time integration:
        - Always performs prediction to time t, updating x̂_{i|i-1} and P_{i|i-1}
        - If t - last_update_time < dt: returns predicted state x̂_{i|i-1} only
        - If t - last_update_time >= dt: performs measurement update to get x̂_{k|k}, updates last_update_time

        The prediction integrates:
            dx̂/dt = A @ x̂ + B @ u
            dP/dt = A @ P + P @ A^T + Q

        Args:
            y: Current measurement vector (n_measurements,) - used only when dt has elapsed
            u: Current input vector (n_inputs,)
            t: Current time (must be non-decreasing between calls)

        Returns:
            np.ndarray: Current state vector estimation [positions, velocities] (n_states,)
                       - Predicted state x̂_{i|i-1} if measurement update is not due
                       - Updated state x̂_{k|k} if measurement update is performed

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

        # Validate time type
        if not isinstance(t, (int, float)):
            raise TypeError(f"Time t must be numeric, got {type(t)}")

        # Always perform prediction to current time
        self._predict(u, t)

        # Calculate time since last update
        time_since_update = t - self.last_update_time

        # Decide whether to perform measurement update
        # Update if: (1) first call, or (2) dt time has elapsed since last update
        if self._first_call or time_since_update >= self.dt:
            # Time for measurement update
            self._update(y)
            self.last_update_time = t  # Update to current time
            self._first_call = False  # No longer the first call
            return self.x_est.copy()  # Return updated state x̂_{k|k}
        else:
            # Return predicted state without update
            return self.x_pred.copy()  # Return predicted state x̂_{i|i-1}
