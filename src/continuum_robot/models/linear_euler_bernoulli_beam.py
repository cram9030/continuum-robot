from enum import Enum
from typing import Dict, Set
import numpy as np
import pandas as pd
from scipy import sparse
from typing import Union
from scipy.sparse.linalg import inv
from scipy.linalg import eig
import pathlib
import warnings


class BoundaryConditionType(Enum):
    """Enumeration of supported boundary condition types."""

    FIXED = "fixed"  # Both displacement and rotation fixed
    PINNED = "pinned"  # Displacement fixed, rotation free


class LinearEulerBernoulliBeam:
    """
    A class implementing the Linear Euler-Bernoulli Beam Model.

    This class creates and manipulates mass and stiffness matrices for a beam
    discretized into finite elements, where each element can have different
    material and geometric properties.

    The matrices are stored internally as sparse matrices but returned as dense
    matrices through getter functions.

    Attributes:
        parameters (pd.DataFrame): Dataframe containing beam section parameters
        K (sparse.csr_matrix): Global stiffness matrix in sparse format
        M (sparse.csr_matrix): Global mass matrix in sparse format
    """

    def __init__(
        self, parameters: Union[str, pathlib.Path, pd.DataFrame], damping_ratio: float
    ):
        """
        Initialize beam with parameters from CSV file.

        Args:
            filename: Path to CSV file containing beam parameters

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """

        if not 0 <= damping_ratio <= 1:
            raise ValueError("Damping ratio must be between 0 and 1")

        self.parameters = None
        self.K = None
        self.M = None
        self.C = None
        self.damping_ratio = damping_ratio

        if isinstance(parameters, (str, pathlib.Path)):
            # Load from CSV
            try:
                self.parameters = pd.read_csv(parameters)
            except FileNotFoundError:
                raise FileNotFoundError(f"Parameter file {parameters} not found")
        elif isinstance(parameters, pd.DataFrame):
            # Use DataFrame directly
            self.parameters = parameters.copy()
        else:
            raise TypeError("Parameters must be filepath or pandas DataFrame")

        # Validate parameters
        self.validate_parameters(self.parameters)

        self._boundary_conditions: Dict[int, BoundaryConditionType] = {}
        self._boundary_conditions_applied = False
        self._constrained_dofs: Set[int] = set()  # Track constrained DOFs

        # Create matrices
        self.create_stiffness_matrix()
        self.create_mass_matrix()
        self.create_damping_matrix()

    def validate_parameters(self, parameters: pd.DataFrame) -> None:
        """
        Validate beam parameters from DataFrame.

        Args:
            parameters: DataFrame containing beam parameters

        Raises:
            ValueError: If parameters are invalid or physically impossible
        """
        required_cols = [
            "length",
            "elastic_modulus",
            "moment_inertia",
            "density",
            "cross_area",
        ]

        # Check required columns exist
        if not all(col in parameters.columns for col in required_cols):
            raise ValueError(
                f"DataFrame must contain columns: {', '.join(required_cols)}"
            )

        # Check for physical validity
        if (parameters[required_cols] <= 0).any().any():
            raise ValueError("All parameters must be positive")

    def update_parameters(self, parameters: pd.DataFrame) -> None:
        """
        Update beam parameters and recreate matrices.

        Args:
            parameters: DataFrame containing new beam parameters
        """
        self.validate_parameters(parameters)
        self.parameters = parameters.copy()
        self.create_stiffness_matrix()
        self.create_mass_matrix()
        self.create_damping_matrix()

    def read_parameter_file(self, filename: Union[str, pathlib.Path]) -> None:
        """
        Read and validate beam parameters from CSV file.

        Args:
            filename: Path to CSV file containing lengths, elastic moduli,
                     moments of inertia, densities, and cross-sectional areas

        Raises:
            ValueError: If parameters are invalid or physically impossible
            FileNotFoundError: If file cannot be found
        """
        try:
            df = pd.read_csv(filename)
            required_cols = [
                "length",
                "elastic_modulus",
                "moment_inertia",
                "density",
                "cross_area",
            ]

            if not all(col in df.columns for col in required_cols):
                raise ValueError(
                    f"CSV must contain columns: {', '.join(required_cols)}"
                )

            # Check for physical validity
            if (df[required_cols] <= 0).any().any():
                raise ValueError("All parameters must be positive")

            self.parameters = df
            # Reset matrices since parameters changed
            self.K = None
            self.M = None
            self.C = None

        except FileNotFoundError:
            raise FileNotFoundError(f"Parameter file {filename} not found")

    def create_stiffness_matrix(self) -> None:
        """
        Create global stiffness matrix K.

        Creates sparse matrix based on beam parameters and stores internally.
        """
        n_segments = len(self.parameters)
        matrix_size = 2 * (n_segments + 1)

        # Lists for sparse matrix construction
        rows, cols, data = [], [], []

        for i in range(n_segments):
            k_local = self._calculate_segment_stiffness(i)
            # Add local matrix entries to global matrix
            for local_i in range(4):
                for local_j in range(4):
                    global_i = 2 * i + local_i
                    global_j = 2 * i + local_j
                    rows.append(global_i)
                    cols.append(global_j)
                    data.append(k_local[local_i, local_j])

        self.K = sparse.csr_matrix(
            (data, (rows, cols)), shape=(matrix_size, matrix_size)
        )

    def create_mass_matrix(self) -> None:
        """
        Create global mass matrix M.

        Creates sparse matrix based on beam parameters and stores internally.
        """
        n_segments = len(self.parameters)
        matrix_size = 2 * (n_segments + 1)

        # Lists for sparse matrix construction
        rows, cols, data = [], [], []

        for i in range(n_segments):
            m_local = self._calculate_segment_mass(i)
            # Add local matrix entries to global matrix
            for local_i in range(4):
                for local_j in range(4):
                    global_i = 2 * i + local_i
                    global_j = 2 * i + local_j
                    rows.append(global_i)
                    cols.append(global_j)
                    data.append(m_local[local_i, local_j])

        self.M = sparse.csr_matrix(
            (data, (rows, cols)), shape=(matrix_size, matrix_size)
        )

    def create_damping_matrix(self) -> None:
        """
        Create global damping matrix C using modal damping approach.

        Creates sparse matrix based on beam parameters and stores internally.
        Uses the relationship C = 2MXξΩX^(-1) where:
        - ξ is the damping ratio times identity matrix
        - M is the mass matrix
        - X contains eigenvectors of M^(-1)K
        - Ω is natural frequencies times identity matrix

        Raises:
            RuntimeError: If K or M matrices haven't been created
            ValueError: If eigenvalue computation fails or produces invalid results
        """

        if self.K is None or self.M is None:
            raise RuntimeError(
                "Matrices must be created before applying boundary conditions"
            )

        MInv = inv(self.M)
        try:
            eigenvalues, eigenvectors = eig(MInv.dot(self.K).toarray())
        except np.linalg.LinAlgError:
            raise ValueError("Failed to compute eigenvalues")

        # Create diagonal matrices
        n_states = self.M.shape[0]
        omega = sparse.diags(np.sqrt(eigenvalues))
        dampting_matrix = sparse.eye(n_states) * self.damping_ratio

        # Convert eigenvectors to sparse
        X = sparse.csr_matrix(eigenvectors)
        X_inv = sparse.csr_matrix(np.linalg.inv(eigenvectors))

        # Compute damping matrix maintaining sparsity
        self.C = 2 * self.M.dot(X.dot(dampting_matrix.dot(omega.dot(X_inv))))

    def get_stiffness_matrix(self) -> np.ndarray:
        """
        Return global stiffness matrix as dense matrix.

        Returns:
            np.ndarray: Global stiffness matrix

        Raises:
            RuntimeError: If matrix hasn't been created
        """
        if self.K is None:
            raise RuntimeError("Stiffness matrix not yet created")
        return self.K.toarray()

    def get_mass_matrix(self) -> np.ndarray:
        """
        Return global mass matrix as dense matrix.

        Returns:
            np.ndarray: Global mass matrix

        Raises:
            RuntimeError: If matrix hasn't been created
        """
        if self.M is None:
            raise RuntimeError("Mass matrix not yet created")
        return self.M.toarray()

    def get_mass_damping(self) -> np.ndarray:
        """
        Return global damping matrix as dense matrix.

        Returns:
            np.ndarray: Global damping matrix

        Raises:
            RuntimeError: If matrix hasn't been created
        """
        if self.C is None:
            raise RuntimeError("Damping matrix not yet created")
        return self.C.toarray()

    def get_length(self) -> float:
        """
        Return total length of beam.

        Returns:
            float: Total beam length
        """
        return self.parameters["length"].sum()

    def get_segment_stiffness(self, i: int) -> np.ndarray:
        """
        Return stiffness matrix for specified segment.

        Args:
            i: Segment index

        Returns:
            np.ndarray: Local stiffness matrix for segment

        Raises:
            IndexError: If segment index is invalid
        """
        if i < 0 or i >= len(self.parameters):
            raise IndexError(f"Segment index {i} out of range")
        return self._calculate_segment_stiffness(i)

    def get_segment_mass(self, i: int) -> np.ndarray:
        """
        Return mass matrix for specified segment.

        Args:
            i: Segment index

        Returns:
            np.ndarray: Local mass matrix for segment

        Raises:
            IndexError: If segment index is invalid
        """
        if i < 0 or i >= len(self.parameters):
            raise IndexError(f"Segment index {i} out of range")
        return self._calculate_segment_mass(i)

    def _calculate_segment_stiffness(self, i: int) -> np.ndarray:
        """Calculate local stiffness matrix for segment i."""
        row = self.parameters.iloc[i]
        L = row["length"]
        EI = row["elastic_modulus"] * row["moment_inertia"]

        return np.array(
            [
                [12 / (L**2), 6 / L, -12 / (L**2), 6 / L],
                [6 / L, 4, -6 / L, 2],
                [-12 / (L**2), -6 / L, 12 / (L**2), -6 / L],
                [6 / L, 2, -6 / L, 4],
            ]
        ) * (EI / L)

    def _calculate_segment_mass(self, i: int) -> np.ndarray:
        """Calculate local mass matrix for segment i."""
        row = self.parameters.iloc[i]
        L = row["length"]
        rhoA = row["density"] * row["cross_area"]

        return np.array(
            [
                [156, -22 * L, 54, 13 * L],
                [-22 * L, 4 * L**2, -13 * L, -3 * L**2],
                [54, -13 * L, 156, 22 * L],
                [13 * L, -3 * L**2, 22 * L, 4 * L**2],
            ]
        ) * (rhoA * L / 420)

    def apply_boundary_conditions(
        self, conditions: Dict[int, BoundaryConditionType]
    ) -> None:
        """
        Apply multiple boundary conditions.

        Args:
            conditions: Dictionary mapping node indices to boundary condition types

        Raises:
            ValueError: If any node index is invalid
            RuntimeError: If matrices haven't been created yet
        """
        if self.K is None or self.M is None:
            raise RuntimeError(
                "Matrices must be created before applying boundary conditions"
            )
        if self.C is None:
            warnings.warn(
                "Warning: Matrix C is None. Proceeding without it.", RuntimeWarning
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
            if bc_type == BoundaryConditionType.FIXED:
                dofs_to_constrain.add(2 * node_idx)  # Displacement
                dofs_to_constrain.add(2 * node_idx + 1)  # Rotation
            elif bc_type == BoundaryConditionType.PINNED:
                dofs_to_constrain.add(2 * node_idx)  # Only displacement
            else:
                raise ValueError(f"Unsupported boundary condition type: {bc_type}")

            self._boundary_conditions[node_idx] = bc_type

        # Get list of DOFs to keep
        all_dofs = set(range(self.M.shape[0]))
        dofs_to_keep = sorted(list(all_dofs - dofs_to_constrain))

        if not dofs_to_keep:  # All DOFs constrained
            raise ValueError("Cannot constrain all degrees of freedom")

        # Create reduced matrices using sparse operations
        self.K = self.K[dofs_to_keep, :][:, dofs_to_keep]
        self.M = self.M[dofs_to_keep, :][:, dofs_to_keep]
        if self.C is not None:
            self.C = self.C[dofs_to_keep, :][:, dofs_to_keep]

        self._constrained_dofs = dofs_to_constrain
        self._boundary_conditions_applied = True

    def clear_boundary_conditions(self) -> None:
        """
        Clear all boundary conditions and recreate original matrices.

        Raises:
            RuntimeError: If original matrices haven't been created
        """
        if self.K is None or self.M is None:
            raise RuntimeError(
                "Matrices must be created before clearing boundary conditions"
            )

        # Recreate original matrices
        self.create_stiffness_matrix()
        self.create_mass_matrix()
        self.create_damping_matrix()

        # Clear stored boundary conditions
        self._boundary_conditions.clear()
        self._constrained_dofs.clear()
        self._boundary_conditions_applied = False

    def get_boundary_conditions(self) -> Dict[int, BoundaryConditionType]:
        """
        Get currently applied boundary conditions.

        Returns:
            Dictionary mapping node indices to their boundary conditions
        """
        return self._boundary_conditions.copy()

    def get_constrained_dofs(self) -> Set[int]:
        """
        Get indices of constrained degrees of freedom.

        Returns:
            Set of constrained DOF indices
        """
        return self._constrained_dofs.copy()

    def has_boundary_conditions(self) -> bool:
        """
        Check if boundary conditions have been applied.

        Returns:
            True if boundary conditions have been applied, False otherwise
        """
        return self._boundary_conditions_applied
