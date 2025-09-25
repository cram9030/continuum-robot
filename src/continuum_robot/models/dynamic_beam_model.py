from typing import Callable, Dict, Union
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import inv
import pathlib

from .abstractions import ElementType, BoundaryConditionType
from .euler_bernoulli_beam import EulerBernoulliBeam
from .fluid_forces import FluidDragForce
from .gravity_forces import GravityForce
from .force_registry import ForceRegistry, InputRegistry
from .force_params import ForceParams


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
        force_params: ForceParams = None,
    ):
        """
        Initialize dynamic beam model from CSV file.

        Args:
            filename: Path to CSV containing beam parameters and element types
            force_params: Optional force parameters for configuring all force effects
                         (fluid dynamics, gravity, etc.)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If parameters are invalid
        """
        # Set force parameters
        self.force_params = force_params or ForceParams()

        # Read and validate parameters
        self.params = pd.read_csv(filename)
        self._validate_parameters()

        # Create boundary conditions dictionary
        self.boundary_conditions = self._process_boundary_conditions()

        # Always use unified beam model
        self.beam_model = EulerBernoulliBeam(self.params)
        self.beam_model.apply_boundary_conditions(self.boundary_conditions)

        # Store constrained DOFs
        self.constrained_dofs = self.beam_model.get_constrained_dofs()

        # Precompute mass matrix inverse for efficiency
        self.M_inv = inv(self.beam_model.M)

        # Initialize system functions
        self.system_func = None
        self.input_func = None

        # Initialize registries for forces and inputs
        self.force_registry = ForceRegistry()
        self.input_registry = InputRegistry()

        # Initialize state mapping first (needed for force components)
        self._initialize_state_mapping()

        # Auto-register forces based on configuration (after state mapping is ready)
        self._auto_register_forces()

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
        if self.force_params.enable_fluid_effects:
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
        if self.force_params.enable_fluid_effects:
            if self.force_params.fluid_density <= 0:
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

        # Get position mapping from unified beam model
        pos_mapping = self.beam_model.dof_to_node_param
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

    def _auto_register_forces(self) -> None:
        """Auto-register force components based on beam configuration."""
        # Register fluid drag forces if enabled
        if self.force_params.enable_fluid_effects:
            fluid_data = self.params[["wetted_area", "drag_coef"]]
            fluid_force = FluidDragForce(
                fluid_data=fluid_data,
                state_mapping=self.state_to_node_param,
                fluid_density=self.force_params.fluid_density,
                enabled=True,
            )
            self.force_registry.register(fluid_force)

        # Register gravity forces if enabled
        if self.force_params.enable_gravity_effects:
            beam_data = self.params[["density", "cross_area", "length"]]
            gravity_force = GravityForce(
                beam_params=beam_data,
                gravity_vector=self.force_params.get_gravity_vector(),
                enabled=True,
            )
            self.force_registry.register(gravity_force)

    def create_system_func(self, forces_func: Callable = None) -> None:
        """Create system matrix function A(x) using unified beam model.

        Args:
            forces_func: Optional external force function that computes forces(x, t).
                        If None, uses forces from the internal force registry.
        """
        M_inv = self.M_inv

        # Use provided forces function or create from registry
        if forces_func is None:
            forces_func = self.force_registry.create_aggregated_function()

        def system(x):
            n_states = len(x) // 2
            positions = x[:n_states]
            velocities = x[n_states:]

            # Calculate stiffness forces using unified beam model
            k_x = self.beam_model.get_stiffness_function()(positions)

            # Get additional forces from external function
            additional_forces = forces_func(x, 0.0)

            return np.concatenate(
                [
                    velocities,
                    -M_inv.dot(k_x) + M_inv.dot(additional_forces),
                ]
            )

        self.system_func = system

    def create_input_func(self) -> None:
        """
        Create beam-specific input function that transforms generalized forces to state derivatives.

        This function creates the input transformation B*u where B is the input matrix that maps
        generalized forces (already aggregated externally) through the inverted mass matrix to
        the beam's state space. The beam is only responsible for this transformation, not for
        input aggregation, control logic, or disturbance handling.

        The created function signature is (x, u, t) -> state_derivatives where:
        - x: current state vector [positions, velocities]
        - u: generalized force vector (pre-aggregated externally)
        - t: current time (unused by beam, but kept for interface compatibility)

        All control inputs, disturbances, and force aggregation should be handled externally
        and passed in as the final generalized force vector u.
        """

        def input_function(x: np.ndarray, u: np.ndarray, t: float = 0.0) -> np.ndarray:
            """
            Transform generalized forces to state derivatives through beam dynamics.

            Args:
                x: Current state vector [positions, velocities]
                u: Generalized force vector applied to position DOFs (pre-aggregated)
                t: Current time (unused, kept for interface compatibility)

            Returns:
                State derivatives resulting from applied generalized forces

            Raises:
                ValueError: If input dimensions are incompatible with beam model
            """
            if not isinstance(x, np.ndarray) or not isinstance(u, np.ndarray):
                raise ValueError("State and input must be numpy arrays")

            if x.ndim != 1 or u.ndim != 1:
                raise ValueError("State and input must be 1D arrays")

            n = len(x) // 2  # Number of position DOFs

            if len(u) != n:
                raise ValueError(
                    f"Input vector length {len(u)} must match position DOFs {n}. "
                    f"Expected {n}, got {len(u)}"
                )

            # Transform generalized forces to state derivatives: [0; M^-1 * u]
            # Upper block: position derivatives are velocities (handled by system_func)
            # Lower block: velocity derivatives are accelerations = M^-1 * forces
            B = sparse.bmat([[sparse.csr_matrix((n, n))], [self.M_inv]], format="csr")

            return B.dot(u)

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

            return self.system_func(x) + self.input_func(x, force, t)

        return dynamic_system
