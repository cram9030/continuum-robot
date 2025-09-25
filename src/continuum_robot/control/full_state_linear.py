import numpy as np
from .control_abstractions import AbstractInputHandler


class FullStateLinear(AbstractInputHandler):
    """
    Full-state linear input handler that applies a linear transformation to the input.

    This handler modifies the input vector based on the current state using a
    predefined gain matrix. It is useful for implementing state feedback control
    strategies where the input needs to be adjusted according to the system's state.

    The transformation is defined as:
        u_modified = u - K * (x - x_ref)

    """

    def __init__(self, gain_matrix: np.ndarray, enabled: bool = True):
        """
        Initialize full-state linear input handler.

        Args:
            gain_matrix: Gain matrix K for state feedback (shape: [input_dim, state_dim])
            enabled: Whether this input handler is enabled
        """
        if gain_matrix.ndim != 2:
            raise ValueError("Gain matrix must be a 2D array.")
        self.gain_matrix = gain_matrix
        self.enabled = enabled

    def compute_input(self, x: np.ndarray, r: np.ndarray, t: float) -> np.ndarray:
        """
        Compute modified input based on current state and reference input.

        Args:
            x: Current state vector [positions, velocities]
            r: Reference input vector
            t: Current time (not used in this handler)
        Returns:
            Modified input vector
        """

        # Ensure input dimensions are compatible
        if r.ndim != 1:
            raise ValueError("Input vector r must be a 1D array.")
        if x.ndim != 1:
            raise ValueError("State vector x must be a 1D array.")
        if x.shape[0] != r.shape[0]:
            raise ValueError(
                "State vector and refrence vector must have the same length."
            )
        if self.gain_matrix.shape[1] != x.shape[0]:
            raise ValueError(
                "Gain matrix column dimension must match state vector length."
            )

        # Compute state feedback term
        feedback_force = self.gain_matrix @ (r - x)

        return feedback_force

    def is_enabled(self) -> bool:
        """Return True if this input handler is enabled."""
        return self.enabled
