import numpy as np
from .abstractions import AbstractForce


class FluidDynamicsParams:
    """Container for fluid dynamics parameters."""

    def __init__(self, fluid_density: float = 0.0, enable_fluid_effects: bool = False):
        """
        Initialize fluid dynamics parameters.

        Args:
            fluid_density: Density of the fluid medium [kg/m³]
            enable_fluid_effects: Whether to enable fluid dynamics effects
        """
        self.fluid_density = fluid_density
        self.enable_fluid_effects = enable_fluid_effects

    def __bool__(self) -> bool:
        """Return True if fluid effects are enabled."""
        return self.enable_fluid_effects


class FluidDragForce(AbstractForce):
    """Fluid drag force implementation for transverse beam motion."""

    def __init__(self, fluid_data, state_mapping, fluid_density, enabled=True):
        """
        Initialize fluid drag force with necessary data for precomputation.

        Args:
            fluid_data: DataFrame with 'wetted_area' and 'drag_coef' columns
            state_mapping: Dictionary mapping state indices to (parameter, node) pairs
            fluid_density: Fluid density for drag force calculations
            enabled: Whether this force component is enabled
        """
        self.fluid_data = fluid_data
        self.state_mapping = state_mapping
        self.fluid_density = fluid_density
        self.enabled = enabled
        self.fluid_coefficients = None

        if self.is_enabled():
            self._precompute_fluid_coefficients()

    def is_enabled(self) -> bool:
        """Return True if fluid effects are enabled."""
        return self.enabled

    def _precompute_fluid_coefficients(self) -> None:
        """Precompute fluid dynamics coefficients using state mapping."""
        if not self.is_enabled():
            return

        # Get wetted areas and drag coefficients from fluid data
        wetted_areas = self.fluid_data["wetted_area"].values
        drag_coefs = self.fluid_data["drag_coef"].values

        # Add one more for final node (use last segment values)
        wetted_areas = np.append(wetted_areas, wetted_areas[-1])
        drag_coefs = np.append(drag_coefs, drag_coefs[-1])

        n_nodes = len(wetted_areas)

        # Dictionary to map nodes to their 'dw_dt' state indices
        node_to_dw_dt_idx = {}
        # Dictionary to map nodes to their 'w' state indices
        node_to_w_idx = {}

        # Find all transverse velocity 'dw_dt' parameters and their corresponding 'w' positions
        for idx, (param, node) in self.state_mapping.items():
            if param == "dw_dt" and node < n_nodes:
                node_to_dw_dt_idx[node] = idx
            elif param == "w" and node < n_nodes:
                node_to_w_idx[node] = idx

        # Build arrays of corresponding indices and drag factors
        w_vel_indices = []  # Indices in state vector for dw_dt
        w_pos_indices = []  # Corresponding indices for w positions
        drag_factors = []  # Drag factor for each node

        # Only include nodes that have both position and velocity state entries
        for node in sorted(set(node_to_dw_dt_idx.keys()) & set(node_to_w_idx.keys())):
            if node < len(wetted_areas) and node < len(drag_coefs):
                w_vel_indices.append(node_to_dw_dt_idx[node])
                w_pos_indices.append(node_to_w_idx[node])
                drag_factor = (
                    0.5 * self.fluid_density * drag_coefs[node] * wetted_areas[node]
                )
                drag_factors.append(drag_factor)

        # Number of position/velocity states
        n_pos_states = len(self.state_mapping) // 2

        # Store the computed coefficients
        self.fluid_coefficients = {
            "w_vel_indices": w_vel_indices,  # Indices of 'dw_dt' velocities in state vector
            "w_pos_indices": w_pos_indices,  # Indices of 'w' positions in state vector
            "drag_factors": drag_factors,  # Drag factors for each node
            "n_pos_states": n_pos_states,  # Number of position states
        }

    def compute_forces(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute fluid drag forces based on current state.

        Args:
            x: Current state vector [positions, velocities]
            t: Current time (unused for drag forces)

        Returns:
            Force vector corresponding to position states
        """
        if not self.is_enabled() or self.fluid_coefficients is None:
            n_states = len(x) // 2
            return np.zeros(n_states)

        n_states = len(x) // 2

        # Get precomputed coefficients
        w_vel_indices = self.fluid_coefficients["w_vel_indices"]
        w_pos_indices = self.fluid_coefficients["w_pos_indices"]
        drag_factors = self.fluid_coefficients["drag_factors"]

        # Create drag forces vector (zeros initially)
        drag_forces = np.zeros(n_states)

        # Apply drag forces based on transverse velocities
        for i in range(len(w_vel_indices)):
            if i < len(drag_factors) and i < len(w_pos_indices):
                vel_idx = w_vel_indices[i]
                pos_idx = w_pos_indices[i]
                # Calculate relative position in velocities array
                force_idx = pos_idx
                vel = x[vel_idx]
                drag_factor = drag_factors[i]
                # Nonlinear drag proportional to velocity^2
                drag_force = -drag_factor * vel * np.abs(vel)
                # Apply force at the position index
                drag_forces[force_idx] = drag_force

        return drag_forces
