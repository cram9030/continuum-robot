import numpy as np
from .control_abstractions import AbstractInputHandler


class FullStateLinear(AbstractInputHandler):
    """
    Full-state linear feedback controller that computes force vectors.

    This handler computes feedback forces based on the current state and reference
    state using a gain matrix and input matrix. It is useful for implementing state
    feedback control strategies such as LQR control.

    For continuous-time systems:
        The feedback force is computed as:
        feedback_force = B @ K @ (r - x)

    For discrete-time systems:
        The control input is computed at discrete intervals and held constant
        between updates:
        u[k] = -K @ (x[k] - r[k])
        feedback_force = B @ u[k]

    where:
        - B is the input matrix mapping control inputs to forces
        - K is the feedback gain matrix
        - r is the reference state
        - x is the current state

    Attributes:
        gain_matrix: Control gain matrix K
        B: Input matrix for force computation
        enabled: Whether this handler is enabled
        ndof: Number of degrees of freedom (state_dim / 2)
        is_discrete: Whether using discrete-time control
        dt: Time step for discrete updates (if discrete)
        last_update_time: Time of last discrete update
        last_control_input: Last computed control input (for discrete systems)
    """

    def __init__(
        self,
        gain_matrix: np.ndarray,
        B_matrix: np.ndarray,
        enabled: bool = True,
        is_discrete: bool = False,
        dt: float = None,
    ):
        """
        Initialize full-state linear input handler.

        Args:
            gain_matrix: Gain matrix K for state feedback (shape: [input_dim, state_dim])
            B_matrix: Input matrix B for force computation (shape: [state_dim, input_dim])
            enabled: Whether this input handler is enabled
            is_discrete: Whether to use discrete-time control (default: False for continuous)
            dt: Time step for discrete updates (required if is_discrete=True)

        Raises:
            ValueError: If matrix dimensions are invalid or dt is not provided for discrete systems
        """
        if gain_matrix.ndim != 2:
            raise ValueError("Gain matrix must be a 2D array.")
        if B_matrix.ndim != 2:
            raise ValueError("B matrix must be a 2D array.")
        if B_matrix.shape[1] != gain_matrix.shape[0]:
            raise ValueError(
                "B matrix column dimension must match gain matrix row dimension."
            )
        if is_discrete and dt is None:
            raise ValueError("dt must be provided for discrete-time control")
        if is_discrete and dt <= 0:
            raise ValueError("dt must be positive for discrete-time control")

        self.gain_matrix = gain_matrix
        self.B = B_matrix
        self.enabled = enabled
        self.ndof = B_matrix.shape[0] // 2
        self.is_discrete = is_discrete
        self.dt = dt
        self.last_update_time = None
        self.last_control_input = None

    def compute_input(self, x: np.ndarray, r: np.ndarray, t: float) -> np.ndarray:
        """
        Compute feedback force based on current state and reference state.

        This method assumes it's called at the appropriate time intervals.
        For discrete-time systems, it should be called at every dt interval.
        For continuous-time systems, it can be called at any time.

        Args:
            x: Current state vector [positions, velocities]
            r: Reference state vector
            t: Current time (unused but part of interface)
        Returns:
            Force vector computed as: B @ u where u = -K @ (x - r)
        """

        # Ensure input dimensions are compatible
        if r.ndim != 1:
            raise ValueError("Reference vector r must be a 1D array.")
        if x.ndim != 1:
            raise ValueError("State vector x must be a 1D array.")
        if x.shape[0] != r.shape[0]:
            raise ValueError(
                "State vector and reference vector must have the same length."
            )
        if self.gain_matrix.shape[1] != x.shape[0]:
            raise ValueError(
                "Gain matrix column dimension must match state vector length."
            )

        # Compute control input: u = -K @ (x - r)
        # Note: We use (x - r) instead of (r - x) because the standard LQR
        # control law is u = -K*x, and we want to regulate to reference r
        control_input = -self.gain_matrix @ (x - r)

        # Map control input to force: feedback_force = B @ u
        # We only use the velocity part of B (lower half)
        feedback_force = self.B[self.ndof :, :] @ control_input

        return feedback_force

    def is_enabled(self) -> bool:
        """Return True if this input handler is enabled."""
        return self.enabled
