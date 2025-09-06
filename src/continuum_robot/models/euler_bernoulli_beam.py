import numpy as np
import pandas as pd
from scipy import sparse
from typing import Dict, Set, Union, Callable, List
import pathlib

from .abstractions import IBeam, ElementType
from .segments import SegmentFactory, create_properties_from_dataframe
from .linear_euler_bernoulli_beam import BoundaryConditionType


class EulerBernoulliBeam(IBeam):
    """
    Unified Euler-Bernoulli beam implementation supporting hybrid beams with mixed
    linear and nonlinear segments.

    This class implements the IBeam interface and can handle beams with any combination
    of linear and nonlinear segments, automatically detecting segment types from the
    'type' column in the parameter DataFrame.
    """

    def __init__(self, parameters: Union[str, pathlib.Path, pd.DataFrame]):
        """
        Initialize unified beam with parameters from CSV file or DataFrame.

        Args:
            parameters: Either path to CSV file or pandas DataFrame containing:
                      - length: Segment lengths
                      - elastic_modulus: Young's modulus values
                      - moment_inertia: Cross-sectional moment of inertia values
                      - density: Material density values
                      - cross_area: Cross-sectional area values
                      - type: Element type ("linear" or "nonlinear")

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If parameters are invalid
        """
        # Load parameters
        if isinstance(parameters, (str, pathlib.Path)):
            try:
                self.parameters = pd.read_csv(parameters)
            except FileNotFoundError:
                raise FileNotFoundError(f"Parameter file {parameters} not found")
        elif isinstance(parameters, pd.DataFrame):
            self.parameters = parameters.copy()
        else:
            raise TypeError("Parameters must be filepath or pandas DataFrame")

        # Validate parameters
        self._validate_parameters()

        # Create segments using factory
        factory = SegmentFactory()
        self.segments = []
        for i in range(len(self.parameters)):
            props = create_properties_from_dataframe(self.parameters, i)
            segment = factory.create_segment(props)
            self.segments.append(segment)

        super().__init__(self.segments)

        # Initialize matrices and functions
        self.M = None
        self.stiffness_func = None

        # Initialize DOF mappings
        self._initialize_dof_mapping()

        # Initialize boundary condition tracking
        self._boundary_conditions: Dict[int, BoundaryConditionType] = {}
        self._boundary_conditions_applied = False
        self._constrained_dofs: Set[int] = set()

        # Create matrices and functions
        self.assemble_mass_matrix()
        self.stiffness_func = self.create_stiffness_function()

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        required_cols = [
            "length",
            "elastic_modulus",
            "moment_inertia",
            "density",
            "cross_area",
            "type",
        ]

        # Check required columns exist
        if not all(col in self.parameters.columns for col in required_cols):
            raise ValueError(
                f"DataFrame must contain columns: {', '.join(required_cols)}"
            )

        # Check for physical validity
        numeric_cols = required_cols[:-1]  # All except 'type'
        if (self.parameters[numeric_cols] <= 0).any().any():
            raise ValueError("All numeric parameters must be positive")

        # Validate element types
        valid_types = {t.value for t in ElementType}
        invalid_types = set(self.parameters["type"].str.lower()) - valid_types
        if invalid_types:
            raise ValueError(f"Invalid element types: {invalid_types}")

    def _initialize_dof_mapping(self):
        """
        Initialize the mapping between DOF indices and (parameter, node) pairs.

        Each node has 3 DOFs: u (axial displacement), w (transverse displacement),
        and phi (rotation). DOF indices are organized as [u1, w1, phi1, u2, w2, phi2, ...]
        """
        n_nodes = len(self.parameters) + 1
        self.dof_to_node_param = {}  # Maps DOF index to (parameter, node) pair
        self.node_param_to_dof = {}  # Maps (parameter, node) pair to DOF index

        for node in range(n_nodes):
            # Axial displacement (u) at node
            self.dof_to_node_param[3 * node] = ("u", node)
            self.node_param_to_dof[("u", node)] = 3 * node

            # Transverse displacement (w) at node
            self.dof_to_node_param[3 * node + 1] = ("w", node)
            self.node_param_to_dof[("w", node)] = 3 * node + 1

            # Rotation (phi) at node
            self.dof_to_node_param[3 * node + 2] = ("phi", node)
            self.node_param_to_dof[("phi", node)] = 3 * node + 2

        # Store original mappings for when boundary conditions are cleared
        self._original_dof_to_node_param = self.dof_to_node_param.copy()
        self._original_node_param_to_dof = self.node_param_to_dof.copy()

    def assemble_mass_matrix(self) -> np.ndarray:
        """Assemble global mass matrix from segments."""
        n_segments = len(self.segments)
        matrix_size = 3 * (n_segments + 1)

        # Lists for sparse matrix construction
        rows, cols, data = [], [], []

        for i, segment in enumerate(self.segments):
            m_local = segment.get_mass_matrix()
            # Add local matrix entries to global matrix
            for local_i in range(6):
                for local_j in range(6):
                    global_i = 3 * i + local_i
                    global_j = 3 * i + local_j
                    rows.append(global_i)
                    cols.append(global_j)
                    data.append(m_local[local_i, local_j])

        self.M = sparse.csr_matrix(
            (data, (rows, cols)), shape=(matrix_size, matrix_size)
        )
        return self.M

    def create_stiffness_function(self) -> Callable:
        """Create global stiffness function from segments."""
        n_segments = len(self.segments)

        # Separate linear and nonlinear segments for efficient handling
        linear_segments = []
        nonlinear_segments = []

        for i, segment in enumerate(self.segments):
            if segment.get_element_type() == ElementType.LINEAR:
                linear_segments.append((i, segment))
            else:
                nonlinear_segments.append((i, segment))

        def global_stiffness_function(x: np.ndarray) -> np.ndarray:
            """
            Compute global stiffness forces from state vector.

            Args:
                x: Global state vector

            Returns:
                Global force vector
            """
            n_nodes = n_segments + 1
            global_forces = np.zeros(3 * n_nodes)

            # Process linear segments (constant stiffness matrices)
            for i, segment in linear_segments:
                start_idx = 3 * i
                segment_state = x[start_idx : start_idx + 6]
                stiffness_matrix = (
                    segment.get_stiffness_func()
                )  # Returns matrix for linear
                segment_forces = stiffness_matrix.dot(segment_state)

                # Assemble into global force vector
                self._assemble_segment_forces(global_forces, segment_forces, i)

            # Process nonlinear segments (state-dependent stiffness functions)
            for i, segment in nonlinear_segments:
                start_idx = 3 * i
                segment_state = x[start_idx : start_idx + 6]
                stiffness_func = (
                    segment.get_stiffness_func()
                )  # Returns function for nonlinear
                segment_forces = stiffness_func(segment_state)

                # Assemble into global force vector
                self._assemble_segment_forces(global_forces, segment_forces, i)

            return global_forces

        return global_stiffness_function

    def _assemble_segment_forces(
        self, global_forces: np.ndarray, segment_forces: np.ndarray, segment_idx: int
    ) -> None:
        """
        Assemble segment forces into global force vector.

        Args:
            global_forces: Global force vector to modify
            segment_forces: Local segment forces [f1_u, f1_w, f1_phi, f2_u, f2_w, f2_phi]
            segment_idx: Index of the segment
        """
        # Map segment forces to global DOFs
        # Node i gets forces [0:3], Node i+1 gets forces [3:6]
        node_1_idx = segment_idx
        node_2_idx = segment_idx + 1

        # Add forces to global vector (segments can share nodes)
        global_forces[3 * node_1_idx : 3 * node_1_idx + 3] += segment_forces[0:3]
        global_forces[3 * node_2_idx : 3 * node_2_idx + 3] += segment_forces[3:6]

    def apply_boundary_conditions(
        self, conditions: Dict[int, BoundaryConditionType]
    ) -> None:
        """Apply boundary conditions to the beam."""
        if self.M is None or self.stiffness_func is None:
            raise RuntimeError(
                "Matrices must be created before applying boundary conditions"
            )

        # Validate all node indices first
        n_nodes = len(self.parameters) + 1
        for node_idx in conditions:
            if node_idx < 0 or node_idx >= n_nodes:
                raise ValueError(f"Node index {node_idx} out of range [0, {n_nodes-1}]")

        # Track DOFs to be constrained
        dofs_to_constrain = set()

        # Process all boundary conditions
        for node_idx, bc_type in conditions.items():
            base_idx = node_idx * 3  # Each node has 3 components [u, w, θ]

            if bc_type == BoundaryConditionType.FIXED:
                # Constrain all DOFs for node
                dofs_to_constrain.add(base_idx)  # u (axial displacement)
                dofs_to_constrain.add(base_idx + 1)  # w (transverse displacement)
                dofs_to_constrain.add(base_idx + 2)  # θ (rotation)
            elif bc_type == BoundaryConditionType.PINNED:
                # Constrain only displacements
                dofs_to_constrain.add(base_idx)  # u (axial displacement)
                dofs_to_constrain.add(base_idx + 1)  # w (transverse displacement)
            else:
                raise ValueError(f"Unsupported boundary condition type: {bc_type}")

            self._boundary_conditions[node_idx] = bc_type

        # Get list of DOFs to keep (unconstrained DOFs)
        all_dofs = set(range(3 * n_nodes))  # Total DOFs = 3 per node
        unconstrained_dofs = sorted(list(all_dofs - dofs_to_constrain))

        if not unconstrained_dofs:  # All DOFs constrained
            raise ValueError("Cannot constrain all degrees of freedom")

        # Apply boundary conditions to mass matrix
        self.M = self.M[unconstrained_dofs, :][:, unconstrained_dofs]

        # Create wrapper around current stiffness function that handles dimension reduction
        original_stiffness = self.stiffness_func

        def stiffness_with_boundary(x_reduced: np.ndarray) -> np.ndarray:
            """
            Wrapper function that maps between reduced and full state spaces.

            Args:
                x_reduced: State vector with constrained DOFs removed

            Returns:
                Reduced force vector with constrained DOFs removed
            """
            # Create full state vector with zeros in constrained DOFs
            x_full = np.zeros(3 * n_nodes)
            for i, dof in enumerate(unconstrained_dofs):
                x_full[dof] = x_reduced[i]

            # Calculate forces using original function
            forces_full = original_stiffness(x_full)

            # Return only unconstrained DOFs
            return forces_full[unconstrained_dofs]

        # Update tracking variables
        self._constrained_dofs = dofs_to_constrain
        self._unconstrained_dofs = unconstrained_dofs  # Store for future use
        self._boundary_conditions_applied = True
        self.stiffness_func = stiffness_with_boundary

        # Update DOF mappings
        self._update_dof_mapping()

    def _update_dof_mapping(self):
        """Update the DOF mappings after boundary conditions are applied."""
        if not self._boundary_conditions_applied:
            return

        # Create new mappings
        new_dof_to_node_param = {}
        new_node_param_to_dof = {}

        # Get all unconstrained DOFs
        unconstrained_dofs = [
            dof
            for dof in sorted(self._original_dof_to_node_param.keys())
            if dof not in self._constrained_dofs
        ]

        # Create new mapping with sequential indices
        for new_idx, old_idx in enumerate(unconstrained_dofs):
            param_node = self._original_dof_to_node_param[old_idx]
            new_dof_to_node_param[new_idx] = param_node
            new_node_param_to_dof[param_node] = new_idx

        # Update the mappings
        self.dof_to_node_param = new_dof_to_node_param
        self.node_param_to_dof = new_node_param_to_dof

    def get_constrained_dofs(self) -> List[int]:
        """Get list of constrained DOF indices."""
        return list(self._constrained_dofs)

    def clear_boundary_conditions(self) -> None:
        """Clear all boundary conditions and recreate original matrices."""
        if self.M is None or self.stiffness_func is None:
            raise RuntimeError(
                "Matrices must be created before clearing boundary conditions"
            )

        # Recreate original matrices
        self.assemble_mass_matrix()
        self.stiffness_func = self.create_stiffness_function()

        # Clear stored boundary conditions
        self._boundary_conditions.clear()
        self._constrained_dofs.clear()
        self._boundary_conditions_applied = False

        # Restore original DOF mappings
        self.dof_to_node_param = self._original_dof_to_node_param.copy()
        self.node_param_to_dof = self._original_node_param_to_dof.copy()

    def get_boundary_conditions(self) -> Dict[int, BoundaryConditionType]:
        """Get currently applied boundary conditions."""
        return self._boundary_conditions.copy()

    def has_boundary_conditions(self) -> bool:
        """Check if boundary conditions have been applied."""
        return self._boundary_conditions_applied

    def get_mass_matrix(self) -> np.ndarray:
        """Return global mass matrix as dense matrix."""
        if self.M is None:
            raise RuntimeError("Mass matrix not yet created")
        return self.M.toarray()

    def get_stiffness_function(self) -> Callable:
        """Return global stiffness function."""
        if self.stiffness_func is None:
            raise RuntimeError("Stiffness function not yet created")
        return self.stiffness_func

    def get_length(self) -> float:
        """Return total length of beam."""
        return self.parameters["length"].sum()

    def get_segment_count(self) -> int:
        """Return number of segments in the beam."""
        return len(self.segments)

    def get_segment_types(self) -> List[ElementType]:
        """Return list of segment types in order."""
        return [segment.get_element_type() for segment in self.segments]

    def is_hybrid(self) -> bool:
        """Check if beam contains both linear and nonlinear segments."""
        types = set(self.get_segment_types())
        return len(types) > 1

    def get_dof_to_node_param(self, dof_idx: int):
        """
        Get the (parameter, node) pair for a given DOF index.

        Args:
            dof_idx: DOF index in the current state vector

        Returns:
            Tuple (parameter, node) where parameter is 'u', 'w', or 'phi'

        Raises:
            KeyError: If the DOF index is invalid
        """
        if dof_idx not in self.dof_to_node_param:
            raise KeyError(f"Invalid DOF index: {dof_idx}")
        return self.dof_to_node_param[dof_idx]

    def get_dof_index(self, node_idx: int, param: str):
        """
        Get the DOF index for a given node and parameter.

        Args:
            node_idx: Node index
            param: Parameter type ('u', 'w', or 'phi')

        Returns:
            DOF index in the current state vector

        Raises:
            KeyError: If the node or parameter is invalid
        """
        if (param, node_idx) not in self.node_param_to_dof:
            raise KeyError(f"Invalid node/parameter combination: ({node_idx}, {param})")
        return self.node_param_to_dof[(param, node_idx)]
