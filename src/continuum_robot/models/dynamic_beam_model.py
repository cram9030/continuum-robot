from typing import Callable, Dict, Union
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import inv
import pathlib
from enum import Enum

from .linear_euler_bernoulli_beam import (
    LinearEulerBernoulliBeam,
    BoundaryConditionType,
)
from .nonlinear_euler_bernoulli_beam import NonlinearEulerBernoulliBeam


class ElementType(Enum):
    """Enumeration of supported element types."""

    LINEAR = "linear"
    NONLINEAR = "nonlinear"


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


class DynamicEulerBernoulliBeam:
    """
    Dynamic model combining linear and nonlinear Euler-Bernoulli beam elements.

    This class creates a dynamic system that can be used with scipy.integrate.solve_ivp.
    The system handles both linear and nonlinear elements defined by their type in the
    input CSV file.
    """

    def __init__(
        self,
        filename: Union[str, pathlib.Path],
        damping_ratio: float = 0.01,
        fluid_params: FluidDynamicsParams = None,
    ):
        """
        Initialize dynamic beam model from CSV file.

        Args:
            filename: Path to CSV containing beam parameters and element types
            damping_ratio: Damping ratio for linear elements [0,1]
            fluid_params: Optional fluid dynamics parameters for simulating beam
                          motion through a fluid medium

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If parameters are invalid
        """
        # Set fluid dynamics parameters
        self.fluid_params = fluid_params or FluidDynamicsParams()

        # Read and validate parameters
        self.params = pd.read_csv(filename)
        self._validate_parameters()

        # Split elements by type
        linear_mask = self.params["type"].str.lower() == ElementType.LINEAR.value
        self.linear_params = self.params[linear_mask].copy()
        self.nonlinear_params = self.params[~linear_mask].copy()

        # Create boundary conditions dictionary
        self.boundary_conditions = self._process_boundary_conditions()

        # Store information about constrained DOFs
        self.linear_constrained_dofs = None
        self.nonlinear_constrained_dofs = None

        # Store mass matrix inverses for efficiency
        self.linear_M_inv = None
        self.nonlinear_M_inv = None

        # Initialize beam models if elements exist
        self.linear_model = None
        if not self.linear_params.empty:
            self.linear_model = LinearEulerBernoulliBeam(
                self.linear_params, damping_ratio
            )
            self.linear_model.apply_boundary_conditions(self.boundary_conditions)

            # Store constrained DOFs
            self.linear_constrained_dofs = self.linear_model.get_constrained_dofs()

            # Precompute mass matrix inverse
            self.linear_M_inv = inv(self.linear_model.M)

        self.nonlinear_model = None
        if not self.nonlinear_params.empty:
            self.nonlinear_model = NonlinearEulerBernoulliBeam(self.nonlinear_params)
            self.nonlinear_model.create_mass_matrix()
            self.nonlinear_model.create_stiffness_function()
            self.nonlinear_model.apply_boundary_conditions(self.boundary_conditions)

            # Store constrained DOFs
            self.nonlinear_constrained_dofs = (
                self.nonlinear_model.get_constrained_dofs()
            )

            # Precompute mass matrix inverse
            self.nonlinear_M_inv = inv(self.nonlinear_model.M)

        # Initialize system functions
        self.system_func = None
        self.input_func = None

        # Initialize state mapping
        self._initialize_state_mapping()

        # Precompute fluid dynamics coefficients if enabled
        self.fluid_coefficients = None
        if self.fluid_params.enable_fluid_effects:
            self._precompute_fluid_coefficients()

    def _validate_parameters(self) -> None:
        """Validate input parameters and types."""
        required_cols = [
            "length",
            "elastic_modulus",
            "moment_inertia",
            "density",
            "cross_area",
            "type",
            "boundary_condition",
        ]

        # Add fluid-specific columns if fluid effects are enabled
        if self.fluid_params.enable_fluid_effects:
            required_cols.extend(["wetted_area", "drag_coef"])

        if not all(col in self.params.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")

        # Validate element types
        valid_types = {t.value for t in ElementType}
        invalid_types = set(self.params["type"].str.lower()) - valid_types
        if invalid_types:
            raise ValueError(f"Invalid element types: {invalid_types}")

        # Validate boundary conditions
        valid_bcs = {"FIXED", "PINNED", "NONE"}
        invalid_bcs = set(self.params["boundary_condition"]) - valid_bcs
        if invalid_bcs:
            raise ValueError(f"Invalid boundary conditions: {invalid_bcs}")

        # Validate fluid parameters if enabled
        if self.fluid_params.enable_fluid_effects:
            if self.fluid_params.fluid_density <= 0:
                raise ValueError("Fluid density must be positive")

            # Validate drag coefficients in data
            if (self.params["drag_coef"] < 0).any():
                raise ValueError("Drag coefficients cannot be negative")

            # Validate wetted areas
            if (self.params["wetted_area"] < 0).any():
                raise ValueError("Wetted areas cannot be negative")

    def _initialize_state_mapping(self):
        """
        Initialize mapping between state vector indices and physical quantities.

        The state vector consists of two parts:
        - First half: positions (u, w, phi for both linear and nonlinear)
        - Second half: velocities (du_dt, dw_dt, dphi_dt for both linear and nonlinear)
        """
        self.state_to_node_param = {}  # Maps state index to (parameter, node) pair
        self.node_param_to_state = {}  # Maps (parameter, node) pair to state index

        if self.linear_model:
            # Get position mapping from linear model
            pos_mapping = self.linear_model.dof_to_node_param
            n_dofs = len(pos_mapping)

            # Create position part of mapping
            for dof_idx, (param, node) in pos_mapping.items():
                self.state_to_node_param[dof_idx] = (param, node)
                self.node_param_to_state[(param, node)] = dof_idx

            # Create velocity part of mapping
            for dof_idx, (param, node) in pos_mapping.items():
                vel_idx = dof_idx + n_dofs
                vel_param = f"d{param}_dt"  # e.g., "w" becomes "dw_dt"
                self.state_to_node_param[vel_idx] = (vel_param, node)
                self.node_param_to_state[(vel_param, node)] = vel_idx

        elif self.nonlinear_model:
            # Get position mapping from nonlinear model
            pos_mapping = self.nonlinear_model.dof_to_node_param
            n_dofs = len(pos_mapping)

            # Create position part of mapping
            for dof_idx, (param, node) in pos_mapping.items():
                self.state_to_node_param[dof_idx] = (param, node)
                self.node_param_to_state[(param, node)] = dof_idx

            # Create velocity part of mapping
            for dof_idx, (param, node) in pos_mapping.items():
                vel_idx = dof_idx + n_dofs
                vel_param = f"d{param}_dt"  # e.g., "u" becomes "du_dt"
                self.state_to_node_param[vel_idx] = (vel_param, node)
                self.node_param_to_state[(vel_param, node)] = vel_idx

        # Store original mappings for when boundary conditions are cleared/reapplied
        self._original_state_to_node_param = self.state_to_node_param.copy()
        self._original_node_param_to_state = self.node_param_to_state.copy()

    def get_state_to_node_param(self, state_idx):
        """
        Get the (parameter, node) pair for a given state index.

        Args:
            state_idx: State vector index

        Returns:
            Tuple (parameter, node)

        Raises:
            KeyError: If the state index is invalid
        """
        if state_idx not in self.state_to_node_param:
            raise KeyError(f"Invalid state index: {state_idx}")
        return self.state_to_node_param[state_idx]

    def get_state_index(self, node_idx, param):
        """
        Get the state vector index for a given node and parameter.

        Args:
            node_idx: Node index
            param: Parameter type ('u', 'w', 'phi', 'du_dt', 'dw_dt', 'dphi_dt', etc.)

        Returns:
            State vector index

        Raises:
            KeyError: If the node/parameter combination is invalid
        """
        if (param, node_idx) not in self.node_param_to_state:
            raise KeyError(f"Invalid node/parameter combination: ({node_idx}, {param})")

        return self.node_param_to_state[(param, node_idx)]

    def get_state_mapping(self):
        """
        Get the complete state vector mapping dictionary.

        Returns:
            Dict: Maps state indices to (parameter, node) pairs
        """
        return self.state_to_node_param.copy()

    def get_node_param_mapping(self):
        """
        Get the complete node-parameter mapping dictionary.

        Returns:
            Dict: Maps (parameter, node) pairs to state indices
        """
        return self.node_param_to_state.copy()

    def _process_boundary_conditions(self) -> Dict[int, BoundaryConditionType]:
        """Process boundary conditions from parameters."""
        conditions = {}
        for i, bc in enumerate(self.params["boundary_condition"]):
            if bc == "FIXED":
                conditions[i] = BoundaryConditionType.FIXED
            elif bc == "PINNED":
                conditions[i] = BoundaryConditionType.PINNED

        # Verify not all DOFs are constrained
        if len(conditions) == len(self.params) + 1:
            raise ValueError("Cannot constrain all nodes with boundary conditions")

        return conditions

    def _precompute_fluid_coefficients(self) -> None:
        """Precompute fluid dynamics coefficients using state mapping."""
        if not self.fluid_params.enable_fluid_effects:
            return

        # Get wetted areas and drag coefficients directly from params
        wetted_areas = self.params["wetted_area"].values
        drag_coefs = self.params["drag_coef"].values

        # Add one more for final node (use last segment values)
        wetted_areas = np.append(wetted_areas, wetted_areas[-1])
        drag_coefs = np.append(drag_coefs, drag_coefs[-1])

        n_nodes = len(wetted_areas)

        # Dictionary to map nodes to their 'dw_dt' state indices
        node_to_dw_dt_idx = {}
        # Dictionary to map nodes to their 'w' state indices
        node_to_w_idx = {}

        # Find all transverse velocity 'dw_dt' parameters and their corresponding 'w' positions
        for idx, (param, node) in self.state_to_node_param.items():
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
                    0.5
                    * self.fluid_params.fluid_density
                    * drag_coefs[node]
                    * wetted_areas[node]
                )
                drag_factors.append(drag_factor)

        # Number of position/velocity states
        n_pos_states = len(self.state_to_node_param) // 2

        # Store the computed coefficients
        self.fluid_coefficients = {
            "w_vel_indices": w_vel_indices,  # Indices of 'dw_dt' velocities in state vector
            "w_pos_indices": w_pos_indices,  # Indices of 'w' positions in state vector
            "drag_factors": drag_factors,  # Drag factors for each node
            "n_pos_states": n_pos_states,  # Number of position states
        }

    def create_system_func(self) -> None:
        """Create system matrix function A(x).

        Creates specialized system function based on model type:
        - Linear only: Uses standard state-space form
        - Nonlinear only: Uses nonlinear stiffness function
        - Mixed is not currently supported

        Raises:
            ValueError: If both linear and nonlinear models exist
        """
        if self.linear_model and self.nonlinear_model:
            raise ValueError("Mixed linear/nonlinear systems not currently supported")

        if self.linear_model:
            # Create linear state space system
            n = self.linear_model.M.shape[0]
            M_inv = self.linear_M_inv
            A = sparse.bmat(
                [
                    [None, sparse.eye(n)],
                    [-M_inv.dot(self.linear_model.K), -M_inv.dot(self.linear_model.C)],
                ],
                format="csr",
            )

            if self.fluid_params.enable_fluid_effects and self.fluid_coefficients:
                # Get precomputed coefficients
                n_pos_states = self.fluid_coefficients["n_pos_states"]
                w_vel_indices = self.fluid_coefficients["w_vel_indices"]
                w_pos_indices = self.fluid_coefficients["w_pos_indices"]
                drag_factors = self.fluid_coefficients["drag_factors"]

                def system_with_fluid(x):
                    # Apply base linear system
                    result = A.dot(x)

                    # Create drag forces vector (zeros initially)
                    drag_forces = np.zeros(n_pos_states)

                    # Apply drag forces based on transverse velocities
                    for i in range(len(w_vel_indices)):
                        if i < len(drag_factors) and i < len(w_pos_indices):
                            vel_idx = w_vel_indices[i]
                            pos_idx = w_pos_indices[i]
                            vel = x[vel_idx]
                            drag_factor = drag_factors[i]
                            # Nonlinear drag proportional to velocity^2
                            drag_force = -drag_factor * vel * np.abs(vel)
                            # Apply force to the position index
                            drag_forces[pos_idx] = drag_force

                    # Add drag forces to acceleration terms through mass matrix inverse
                    result[n_pos_states:] += M_inv.dot(drag_forces)

                    return result

                self.system_func = system_with_fluid
            else:
                self.system_func = lambda x: A.dot(x)

        elif self.nonlinear_model:
            # Create nonlinear system
            M_inv = self.nonlinear_M_inv

            if self.fluid_params.enable_fluid_effects and self.fluid_coefficients:
                # Get precomputed coefficients
                n_pos_states = self.fluid_coefficients["n_pos_states"]
                w_vel_indices = self.fluid_coefficients["w_vel_indices"]
                w_pos_indices = self.fluid_coefficients["w_pos_indices"]
                drag_factors = self.fluid_coefficients["drag_factors"]

                def nonlinear_system_with_fluid(x):
                    n = len(x) // 2
                    positions = x[:n]
                    velocities = x[n:]

                    # Calculate nonlinear stiffness forces
                    k_x = self.nonlinear_model.get_stiffness_function()(positions)

                    # Create drag forces vector (zeros initially)
                    drag_forces = np.zeros_like(velocities)

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
                    # Combine position derivatives (velocities) with
                    # velocity derivatives (accelerations = forces/mass)
                    return np.concatenate(
                        [
                            velocities,
                            -M_inv.dot(k_x) + M_inv.dot(drag_forces),
                        ]
                    )

                self.system_func = nonlinear_system_with_fluid
            else:

                def nonlinear_system(x):
                    n = len(x) // 2
                    positions = x[:n]
                    velocities = x[n:]

                    # Calculate stiffness forces
                    k_x = self.nonlinear_model.get_stiffness_function()(positions)

                    return np.concatenate([velocities, -M_inv.dot(k_x)])

                self.system_func = nonlinear_system
        else:
            raise ValueError("No model types defined")

    def create_input_func(self) -> None:
        """Create input matrix function B(x)."""

        def input_function(x: np.ndarray) -> np.ndarray:
            n = len(x) // 2
            if self.linear_model:
                M_inv = self.linear_M_inv
            else:
                M_inv = self.nonlinear_M_inv

            B = sparse.bmat([[sparse.csr_matrix((n, n))], [M_inv]], format="csr")
            return B

        self.input_func = input_function

    def get_system_func(self) -> Callable:
        """Return current system function."""
        if self.system_func is None:
            raise RuntimeError("System function not yet created")
        return self.system_func

    def get_dynamic_system(self) -> Callable:
        """Return complete dynamic system for solve_ivp."""
        if self.system_func is None or self.input_func is None:
            raise RuntimeError("System and input functions must be created first")

        def dynamic_system(
            t: float, x: np.ndarray, u: Union[np.ndarray, Callable]
        ) -> np.ndarray:
            """Dynamic system function compatible with solve_ivp.

            Args:
                t: Current time
                x: Current state
                u: External force vector or force function of time

            Returns:
                State derivative vector
            """
            # Get force input
            if callable(u):
                force = u(t)
            else:
                force = u

            return self.system_func(x) + self.input_func(x).dot(force)

        return dynamic_system
