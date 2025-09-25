from abc import ABC, abstractmethod
import numpy as np


class AbstractInputHandler(ABC):
    """Abstract base class for input processing components."""

    @abstractmethod
    def compute_input(self, x: np.ndarray, r: np.ndarray, t: float) -> np.ndarray:
        """
        Compute input modifications given current state, input, and time.

        Args:
            x: Current state vector [positions, velocities]
            r: Refrence input vector
            t: Current time

        Returns:
            Input modification vector (delta) to be added to original input
        """
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """Return True if this input handler is enabled."""
        pass
