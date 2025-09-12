import numpy as np
import warnings
from typing import List, Optional
from .abstractions import AbstractForce


class GravityForce(AbstractForce):
    """
    Gravity force component that applies gravitational forces to beam segments.

    This force applies mass-distributed gravity forces to each beam segment
    based on the segment properties (density, cross-sectional area, length).
    The gravity forces are transformed from global to local coordinates using
    the current beam deformation state (rotation angles).

    The 3D gravity vector is mapped as follows:
    - gravity_vector[0] (gx): Global X gravity component
    - gravity_vector[1] (gy): Global Y gravity component
    - gravity_vector[2] (gz): Currently unused (2D beam model)

    Local beam coordinates:
    - Axial direction (u): along the deformed beam axis
    - Transverse direction (w): perpendicular to the deformed beam axis
    """

    def __init__(
        self,
        beam_instance,
        gravity_vector: Optional[List[float]] = None,
        enabled: bool = True,
    ):
        """
        Initialize gravity force component.

        Args:
            beam_instance: Reference to the beam object containing segment properties
            gravity_vector: 3D gravity acceleration vector [gx, gy, gz] (m/s²)
                          Defaults to [0, -9.81, 0] (standard downward gravity)
            enabled: Whether this force component is enabled
        """
        self.beam = beam_instance
        self.gravity_vector = np.array(
            gravity_vector if gravity_vector is not None else [0.0, -9.81, 0.0]
        )
        self.enabled = enabled

        if len(self.gravity_vector) != 3:
            raise ValueError(
                "Gravity vector must have exactly 3 components [gx, gy, gz]"
            )

        # Get segments and pre-compute segment masses
        # For DynamicEulerBernoulliBeam, segments are in beam.beam_model.segments
        if hasattr(self.beam, "beam_model") and hasattr(
            self.beam.beam_model, "segments"
        ):
            segments = self.beam.beam_model.segments
        elif hasattr(self.beam, "segments"):
            segments = self.beam.segments
        else:
            segments = None

        if not segments:
            warnings.warn(
                "Beam instance does not have segments attribute or segments list is empty. "
                "GravityForce requires beam segments to compute mass-distributed forces.",
                UserWarning,
            )
            self._segment_masses = []
        else:
            # Pre-compute segment masses (mass = density * cross_area * length)
            self._segment_masses = []
            for segment in segments:
                props = segment.get_properties()
                segment_mass = props.density * props.cross_area * props.length
                self._segment_masses.append(segment_mass)

    def compute_forces(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute gravitational forces for all segments, accounting for beam deformation.

        Args:
            x: Current state vector [positions, velocities]
            t: Current time (unused for gravity)

        Returns:
            Force vector corresponding to position states

        Raises:
            RuntimeError: If beam does not have segments available
        """
        n_states = len(x) // 2
        forces = np.zeros(n_states)

        # Check that beam has segments
        if not self._segment_masses:
            raise RuntimeError(
                "Cannot compute gravity forces: beam instance does not have segments available "
                "or segment masses were not pre-computed."
            )

        # Extract position states (first half of state vector)
        positions = x[:n_states]

        # Get gravity components
        gx, gy = self.gravity_vector[0], self.gravity_vector[1]

        # Apply gravity forces to each segment
        for i, segment_mass in enumerate(self._segment_masses):
            # Calculate DOF indices for this segment
            # Each segment connects two nodes (i and i+1)
            # Node i has DOFs at [3*i, 3*i+1, 3*i+2] = [u_i, w_i, φ_i]
            # Node i+1 has DOFs at [3*(i+1), 3*(i+1)+1, 3*(i+1)+2]

            # Get rotation angles at both nodes for this segment
            start_phi_idx = 3 * i + 2  # φ_i (rotation at start node)
            end_phi_idx = 3 * (i + 1) + 2  # φ_{i+1} (rotation at end node)

            # Use average rotation angle for the segment
            if start_phi_idx < n_states and end_phi_idx < n_states:
                phi_avg = 0.5 * (positions[start_phi_idx] + positions[end_phi_idx])
            elif start_phi_idx < n_states:
                phi_avg = positions[start_phi_idx]
            elif end_phi_idx < n_states:
                phi_avg = positions[end_phi_idx]
            else:
                phi_avg = 0.0  # No rotation information available

            # Transform gravity from global to local coordinates
            # Local axial direction: cos(φ) * gx + sin(φ) * gy
            # Local transverse direction: -sin(φ) * gx + cos(φ) * gy
            cos_phi = np.cos(phi_avg)
            sin_phi = np.sin(phi_avg)

            local_axial_gravity = cos_phi * gx + sin_phi * gy
            local_transverse_gravity = -sin_phi * gx + cos_phi * gy

            # Calculate segment force contributions (split between two nodes)
            segment_force_axial = local_axial_gravity * segment_mass * 0.5
            segment_force_transverse = local_transverse_gravity * segment_mass * 0.5

            # Apply to start node (node i)
            start_u_idx = 3 * i  # Axial DOF
            start_w_idx = 3 * i + 1  # Transverse DOF

            # Apply to end node (node i+1)
            end_u_idx = 3 * (i + 1)  # Axial DOF
            end_w_idx = 3 * (i + 1) + 1  # Transverse DOF

            # Check bounds and apply forces
            if start_u_idx < n_states:
                forces[start_u_idx] += segment_force_axial
            if start_w_idx < n_states:
                forces[start_w_idx] += segment_force_transverse
            if end_u_idx < n_states:
                forces[end_u_idx] += segment_force_axial
            if end_w_idx < n_states:
                forces[end_w_idx] += segment_force_transverse

        return forces

    def is_enabled(self) -> bool:
        """Return True if this force component is enabled."""
        return self.enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable this force component."""
        self.enabled = enabled

    def set_gravity_vector(self, gravity_vector: List[float]) -> None:
        """
        Update the gravity vector.

        Args:
            gravity_vector: New 3D gravity acceleration vector [gx, gy, gz] (m/s²)
        """
        if len(gravity_vector) != 3:
            raise ValueError(
                "Gravity vector must have exactly 3 components [gx, gy, gz]"
            )
        self.gravity_vector = np.array(gravity_vector)

    def get_gravity_vector(self) -> np.ndarray:
        """Get the current gravity vector."""
        return self.gravity_vector.copy()
