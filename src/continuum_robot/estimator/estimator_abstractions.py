from abc import ABC, abstractmethod
import numpy as np


class AbstractEstimatorHandler(ABC):
    """Abstract base class for estimating states."""

    @abstractmethod
    def estimate_states(self, y: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """
        Estimate current states given observations, input, and time.

        Args:
            y: Current state vector [positions, velocities]
            u: Current input vector
            t: Current time

        Returns:
            x: Current state vector estimation [positions, velocities]
        """
        pass
