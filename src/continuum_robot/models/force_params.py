from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class ForceParams:
    """
    Unified force parameters for DynamicEulerBernoulliBeam.

    This dataclass consolidates all force-related configuration parameters
    for the dynamic beam model, providing a clean and extensible API for
    enabling and configuring different types of forces.
    """

    # Fluid parameters
    fluid_density: float = 0.0
    enable_fluid_effects: bool = False

    # Gravity parameters
    gravity_vector: List[float] = field(default_factory=lambda: [0.0, -9.81, 0.0])
    enable_gravity_effects: bool = False

    def __post_init__(self):
        """Validate and normalize parameters after initialization."""
        # Convert gravity_vector to numpy array for easier manipulation
        self.gravity_vector = np.array(self.gravity_vector, dtype=float)

        # Auto-disable gravity if vector is all zeros
        if np.allclose(self.gravity_vector, [0.0, 0.0, 0.0]):
            self.enable_gravity_effects = False

        # Validate gravity vector length
        if len(self.gravity_vector) != 3:
            raise ValueError(
                "gravity_vector must have exactly 3 components [gx, gy, gz]"
            )

        # Validate fluid parameters
        if self.enable_fluid_effects and self.fluid_density <= 0:
            raise ValueError(
                "fluid_density must be positive when fluid effects are enabled"
            )

    def __bool__(self) -> bool:
        """Return True if any force effects are enabled."""
        return self.enable_fluid_effects or self.enable_gravity_effects

    def get_gravity_vector(self) -> np.ndarray:
        """Get a copy of the gravity vector."""
        return self.gravity_vector.copy()

    def set_gravity_vector(self, gravity_vector: List[float]) -> None:
        """
        Update the gravity vector.

        Args:
            gravity_vector: New 3D gravity acceleration vector [gx, gy, gz] (m/s²)
        """
        if len(gravity_vector) != 3:
            raise ValueError(
                "gravity_vector must have exactly 3 components [gx, gy, gz]"
            )

        self.gravity_vector = np.array(gravity_vector, dtype=float)

        # Auto-disable gravity if new vector is all zeros
        if np.allclose(self.gravity_vector, [0.0, 0.0, 0.0]):
            self.enable_gravity_effects = False
