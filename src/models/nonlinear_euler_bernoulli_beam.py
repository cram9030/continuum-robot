import numpy as np
import pandas as pd
from scipy import sparse
from typing import Dict, Set, Union, Callable
import pathlib
from functools import partial
from enum import Enum


class BoundaryConditionType(Enum):
    """Enumeration of supported boundary condition types."""

    FIXED = "fixed"  # Both displacement and rotation fixed
    PINNED = "pinned"  # Displacement fixed, rotation free


class NonlinearEulerBernoulliBeam:
    """
    A class implementing the Nonlinear Euler-Bernoulli Beam Model.

    This class creates and manipulates mass matrices and nonlinear stiffness functions
    for a beam discretized into finite elements, where each element can have different
    material and geometric properties.

    The nonlinear stiffness is represented as a function K(x) that depends on the
    current state x=[u₁, θ₁, w₁, u₂, θ₂, w₂].

    Attributes:
        parameters (pd.DataFrame): Dataframe containing beam section parameters
        stiffness_func (Callable): Nonlinear stiffness function K(x)
        M (sparse.csr_matrix): Global mass matrix in sparse format
    """

    def __init__(self, parameters: Union[str, pathlib.Path, pd.DataFrame]):
        """
        Initialize beam with parameters from CSV file or DataFrame.

        Args:
            parameters: Either path to CSV file or pandas DataFrame containing:
                    - length: Segment lengths
                    - elastic_modulus: Young's modulus values
                    - moment_inertia: Cross-sectional moment of inertia values
                    - density: Material density values
                    - cross_area: Cross-sectional area values

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If parameters are invalid
        """
        self.parameters = None
        self.stiffness_func = None
        self.M = None

        # Initialize boundary condition tracking
        self._boundary_conditions: Dict[int, BoundaryConditionType] = {}
        self._boundary_conditions_applied = False
        self._constrained_dofs: Set[int] = set()

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
        Update beam parameters with new DataFrame.

        Args:
            parameters: DataFrame containing new beam parameters

        Raises:
            ValueError: If parameters are invalid
        """
        self.validate_parameters(parameters)
        self.parameters = parameters.copy()

        # Reset matrices/functions since parameters changed
        self.stiffness_func = None
        self.M = None

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

            # Reset matrices/functions since parameters changed
            self.stiffness_func = None
            self.M = None

        except FileNotFoundError:
            raise FileNotFoundError(f"Parameter file {filename} not found")

    def _calculate_segment_stiffness_function(self, i: int) -> Callable:
        """
        Calculate nonlinear stiffness function for segment i.

        Args:
            i: Segment index

        Returns:
            Callable that computes segment stiffness given state vector
        """
        row = self.parameters.iloc[i]
        seg_l = row["length"]
        EI = row["elastic_modulus"] * row["moment_inertia"]
        EA = row["elastic_modulus"] * row["cross_area"]

        def stiffness_func(x: np.ndarray) -> np.ndarray:
            # Get states for this segment
            u1, theta1, w1, u2, theta2, w2 = x

            # Implement the nonlinear stiffness function f1-f6
            # from the provided equations
            f1 = partial(_f_1_expr, length=seg_l, A_xx=EA)
            f2 = partial(_f_2_expr, length=seg_l, A_xx=EA)
            f3 = partial(_f_3_expr, length=seg_l, A_xx=EA, D_xx=EI)
            f4 = partial(_f_4_expr, length=seg_l, A_xx=EA, D_xx=EI)
            f5 = partial(_f_5_expr, length=seg_l, A_xx=EA, D_xx=EI)
            f6 = partial(_f_6_expr, length=seg_l, A_xx=EA, D_xx=EI)

            # Evaluate with remaining variables
            return np.array(
                [
                    f1(u1, theta1, w1, u2, theta2, w2),
                    f3(u1, theta1, w1, u2, theta2, w2),
                    f4(u1, theta1, w1, u2, theta2, w2),
                    f2(u1, theta1, w1, u2, theta2, w2),
                    f5(u1, theta1, w1, u2, theta2, w2),
                    f6(u1, theta1, w1, u2, theta2, w2),
                ]
            )

        def _f_1_expr(u1, theta1, w1, u2, theta2, w2, *, length: float, A_xx: float):
            """
            Expression for axial force at node 1.

            Parameters:
            u1 (float): Displacement at node 1.
            theta1 (float): Rotation at node 1.
            w1 (float): Transverse displacement at node 1.
            u2 (float): Displacement at node 2.
            theta2 (float): Rotation at node 2.
            w2 (float): Transverse displacement at node 2.
            length (float): Length of the beam element.
            A_xx (float): Elastic modulus times cross-sectional area (EA).

            Returns:
            float: Axial force at node 1.
            """
            # A_xx is elastic modulus * cross sectional area (EA)
            # D_xx is elastic modulus * moment of inertia (EI)
            return (
                A_xx
                * (
                    length
                    * (
                        -theta1
                        * (
                            0.0666666666666665 * theta1 * length
                            - 0.0166666666666667 * theta2 * length
                            - 0.05 * w1
                            + 0.05 * w2
                        )
                        + theta2
                        * (
                            0.0166666666666667 * theta1 * length
                            - 0.0666666666666667 * theta2 * length
                            + 0.05 * w1
                            - 0.05 * w2
                        )
                        + u1
                    )
                    + (-u2 - w1 + w2)
                    * (
                        -0.05 * theta1 * length
                        - 0.05 * theta2 * length
                        + 0.6 * w1
                        - 0.6 * w2
                    )
                )
                / length**2
            )

        def _f_2_expr(u1, theta1, w1, u2, theta2, w2, *, length: float, A_xx: float):
            """
            Expression for bending force at node 1.

            Parameters:
            u1 (float): Displacement at node 1.
            theta1 (float): Rotation at node 1.
            w1 (float): Transverse displacement at node 1.
            u2 (float): Displacement at node 2.
            theta2 (float): Rotation at node 2.
            w2 (float): Transverse displacement at node 2.
            length (float): Length of the beam element.
            A_xx (float): Elastic modulus times cross-sectional area (EA).

            Returns:
            float: bending force at node 1.
            """
            return (
                A_xx
                * (
                    length
                    * (
                        theta1
                        * (
                            0.0666666666666665 * theta1 * length
                            - 0.0166666666666667 * theta2 * length
                            - 0.05 * w1
                            + 0.05 * w2
                        )
                        - theta2
                        * (
                            0.0166666666666667 * theta1 * length
                            - 0.0666666666666667 * theta2 * length
                            + 0.05 * w1
                            - 0.05 * w2
                        )
                        - u1
                        + u2
                    )
                    + (w1 - w2)
                    * (
                        -0.05 * theta1 * length
                        - 0.05 * theta2 * length
                        + 0.6 * w1
                        - 0.6 * w2
                    )
                )
                / length**2
            )

        def _f_3_expr(
            u1, theta1, w1, u2, theta2, w2, *, length: float, A_xx: float, D_xx: float
        ):
            """
            Expression for bending moment at node 1.

            Parameters:
            u1 (float): Displacement at node 1.
            theta1 (float): Rotation at node 1.
            w1 (float): Transverse displacement at node 1.
            u2 (float): Displacement at node 2.
            theta2 (float): Rotation at node 2.
            w2 (float): Transverse displacement at node 2.
            length (float): Length of the beam element.
            A_xx (float): Elastic modulus times cross-sectional area (EA).

            Returns:
            float: bending moment at node 1.
            """
            return (
                0.0333333333333333
                * (
                    0.107142857143003 * A_xx * theta1**3 * length**3
                    - 0.32142857142901 * A_xx * theta1**2 * theta2 * length**3
                    + 3.857142857143 * A_xx * theta1**2 * length**2 * w1
                    - 3.857142857143 * A_xx * theta1**2 * length**2 * w2
                    - 0.32142857142901 * A_xx * theta1 * theta2**2 * length**3
                    - 4.0 * A_xx * theta1 * length**3 * u2
                    + 3.0 * A_xx * theta1 * length**2 * u1
                    - 11.5714285714239 * A_xx * theta1 * length * w1**2
                    + 23.1428571428478 * A_xx * theta1 * length * w1 * w2
                    - 11.5714285714239 * A_xx * theta1 * length * w2**2
                    + 0.107142857143003 * A_xx * theta2**3 * length**3
                    + 3.857142857143 * A_xx * theta2**2 * length**2 * w1
                    - 3.857142857143 * A_xx * theta2**2 * length**2 * w2
                    + 1.0 * A_xx * theta2 * length**3 * u2
                    + 3.0 * A_xx * theta2 * length**2 * u1
                    - 11.571428571429 * A_xx * theta2 * length * w1**2
                    + 23.142857142858 * A_xx * theta2 * length * w1 * w2
                    - 11.571428571429 * A_xx * theta2 * length * w2**2
                    + 3.0 * A_xx * length**2 * u2 * w1
                    - 3.0 * A_xx * length**2 * u2 * w2
                    - 36.0 * A_xx * length * u1 * w1
                    + 36.0 * A_xx * length * u1 * w2
                    + 30.857142857144 * A_xx * w1**3
                    - 92.5714285714321 * A_xx * w1**2 * w2
                    + 92.5714285714321 * A_xx * w1 * w2**2
                    - 30.857142857144 * A_xx * w2**3
                    - 180.0 * D_xx * theta1 * length
                    - 180.0 * D_xx * theta2 * length
                    + 360.0 * D_xx * w1
                    - 360.0 * D_xx * w2
                )
                / length**3
            )

        def _f_4_expr(
            u1, theta1, w1, u2, theta2, w2, *, length: float, A_xx: float, D_xx: float
        ):
            """
            Expression for axial force at node 2.

            Parameters:
            u1 (float): Displacement at node 1.
            theta1 (float): Rotation at node 1.
            w1 (float): Transverse displacement at node 1.
            u2 (float): Displacement at node 2.
            theta2 (float): Rotation at node 2.
            w2 (float): Transverse displacement at node 2.
            length (float): Length of the beam element.
            A_xx (float): Elastic modulus times cross-sectional area (EA).

            Returns:
            float: axial force at node 2.
            """
            return (
                0.0285714285714391 * A_xx * theta1**3 * length
                - 0.0107142857142861 * A_xx * theta1**2 * theta2 * length
                + 0.0107142857142719 * A_xx * theta1**2 * w1
                - 0.0107142857142719 * A_xx * theta1**2 * w2
                + 0.00714285714286444 * A_xx * theta1 * theta2**2 * length
                - 0.0214285714286007 * A_xx * theta1 * theta2 * w1
                + 0.0214285714286007 * A_xx * theta1 * theta2 * w2
                - 0.133333333333333 * A_xx * theta1 * u1
                + 0.133333333333333 * A_xx * theta1 * u2
                + 0.128571428571433 * A_xx * theta1 * w1**2 / length
                - 0.257142857142867 * A_xx * theta1 * w1 * w2 / length
                + 0.128571428571433 * A_xx * theta1 * w2**2 / length
                - 0.00357142857143344 * A_xx * theta2**3 * length
                - 0.0107142857142719 * A_xx * theta2**2 * w1
                + 0.0107142857142719 * A_xx * theta2**2 * w2
                + 0.0333333333333333 * A_xx * theta2 * u1
                - 0.0333333333333333 * A_xx * theta2 * u2
                + 0.1 * A_xx * u1 * w1 / length
                - 0.1 * A_xx * u1 * w2 / length
                - 0.1 * A_xx * u2 * w1 / length
                + 0.1 * A_xx * u2 * w2 / length
                - 0.128571428571377 * A_xx * w1**3 / length**2
                + 0.38571428571413 * A_xx * w1**2 * w2 / length**2
                - 0.38571428571413 * A_xx * w1 * w2**2 / length**2
                + 0.128571428571377 * A_xx * w2**3 / length**2
                + 4.0 * D_xx * theta1 / length
                + 2.0 * D_xx * theta2 / length
                - 6.0 * D_xx * w1 / length**2
                + 6.0 * D_xx * w2 / length**2
            )

        def _f_5_expr(
            u1, theta1, w1, u2, theta2, w2, *, length: float, A_xx: float, D_xx: float
        ):
            """
            Expression for bending force at node 2.

            Parameters:
            u1 (float): Displacement at node 1.
            theta1 (float): Rotation at node 1.
            w1 (float): Transverse displacement at node 1.
            u2 (float): Displacement at node 2.
            theta2 (float): Rotation at node 2.
            w2 (float): Transverse displacement at node 2.
            length (float): Length of the beam element.
            A_xx (float): Elastic modulus times cross-sectional area (EA).

            Returns:
            float: bending force at node 2.
            """
            return (
                0.1
                * (
                    -0.0357142857143344 * A_xx * theta1**3 * length**3
                    + 0.107142857143003 * A_xx * theta1**2 * theta2 * length**3
                    - 1.28571428571433 * A_xx * theta1**2 * length**2 * w1
                    + 1.28571428571433 * A_xx * theta1**2 * length**2 * w2
                    + 0.107142857143003 * A_xx * theta1 * theta2**2 * length**3
                    - 1.0 * A_xx * theta1 * length**2 * u1
                    + 1.0 * A_xx * theta1 * length**2 * u2
                    + 3.8571428571413 * A_xx * theta1 * length * w1**2
                    - 7.7142857142826 * A_xx * theta1 * length * w1 * w2
                    + 3.8571428571413 * A_xx * theta1 * length * w2**2
                    - 0.0357142857143344 * A_xx * theta2**3 * length**3
                    - 1.28571428571433 * A_xx * theta2**2 * length**2 * w1
                    + 1.28571428571433 * A_xx * theta2**2 * length**2 * w2
                    - 1.0 * A_xx * theta2 * length**2 * u1
                    + 1.0 * A_xx * theta2 * length**2 * u2
                    + 3.857142857143 * A_xx * theta2 * length * w1**2
                    - 7.71428571428601 * A_xx * theta2 * length * w1 * w2
                    + 3.857142857143 * A_xx * theta2 * length * w2**2
                    + 12.0 * A_xx * length * u1 * w1
                    - 12.0 * A_xx * length * u1 * w2
                    - 12.0 * A_xx * length * u2 * w1
                    + 12.0 * A_xx * length * u2 * w2
                    - 10.2857142857147 * A_xx * w1**3
                    + 30.857142857144 * A_xx * w1**2 * w2
                    - 30.857142857144 * A_xx * w1 * w2**2
                    + 10.2857142857147 * A_xx * w2**3
                    + 60.0 * D_xx * theta1 * length
                    + 60.0 * D_xx * theta2 * length
                    - 120.0 * D_xx * w1
                    + 120.0 * D_xx * w2
                )
                / length**3
            )

        def _f_6_expr(
            u1, theta1, w1, u2, theta2, w2, *, length: float, A_xx: float, D_xx: float
        ):
            """
            Expression for bending moment at node 2.

            Parameters:
            u1 (float): Displacement at node 1.
            theta1 (float): Rotation at node 1.
            w1 (float): Transverse displacement at node 1.
            u2 (float): Displacement at node 2.
            theta2 (float): Rotation at node 2.
            w2 (float): Transverse displacement at node 2.
            length (float): Length of the beam element.
            A_xx (float): Elastic modulus times cross-sectional area (EA).

            Returns:
            float: bending moment at node 2.
            """
            return (
                -0.00357142857143344 * A_xx * theta1**3 * length
                + 0.00714285714286356 * A_xx * theta1**2 * theta2 * length
                - 0.0107142857143003 * A_xx * theta1**2 * w1
                + 0.0107142857143003 * A_xx * theta1**2 * w2
                - 0.0107142857142932 * A_xx * theta1 * theta2**2 * length
                - 0.021428571428558 * A_xx * theta1 * theta2 * w1
                + 0.021428571428558 * A_xx * theta1 * theta2 * w2
                + 0.0333333333333333 * A_xx * theta1 * u1
                - 0.0333333333333333 * A_xx * theta1 * u2
                + 0.0285714285714271 * A_xx * theta2**3 * length
                + 0.0107142857142932 * A_xx * theta2**2 * w1
                - 0.0107142857142932 * A_xx * theta2**2 * w2
                - 0.133333333333333 * A_xx * theta2 * u1
                + 0.133333333333333 * A_xx * theta2 * u2
                + 0.128571428571428 * A_xx * theta2 * w1**2 / length
                - 0.257142857142856 * A_xx * theta2 * w1 * w2 / length
                + 0.128571428571428 * A_xx * theta2 * w2**2 / length
                + 0.1 * A_xx * u1 * w1 / length
                - 0.1 * A_xx * u1 * w2 / length
                - 0.1 * A_xx * u2 * w1 / length
                + 0.1 * A_xx * u2 * w2 / length
                - 0.128571428571433 * A_xx * w1**3 / length**2
                + 0.3857142857143 * A_xx * w1**2 * w2 / length**2
                - 0.3857142857143 * A_xx * w1 * w2**2 / length**2
                + 0.128571428571433 * A_xx * w2**3 / length**2
                + 2.0 * D_xx * theta1 / length
                + 4.0 * D_xx * theta2 / length
                - 6.0 * D_xx * w1 / length**2
                + 6.0 * D_xx * w2 / length**2
            )

        return stiffness_func

    def _calculate_segment_mass(self, i: int) -> np.ndarray:
        """Calculate mass matrix for segment i."""
        row = self.parameters.iloc[i]
        L = row["length"]
        rhoA = row["density"] * row["cross_area"]

        return np.array(
            [
                [140, 0, 0, 70, 0, 0],
                [0, 156, 22 * L, 0, 54, -13 * L],
                [0, 22 * L, 4 * L**2, 0, 13 * L, -3 * L**2],
                [70, 0, 0, 140, 0, 0],
                [0, 54, 13 * L, 0, 156, -22 * L],
                [0, -13 * L, -3 * L**2, 0, -22 * L, 4 * L**2],
            ]
        ) * (rhoA * L / 420)

    def create_stiffness_function(self) -> None:
        """Create global nonlinear stiffness function."""
        n_segments = len(self.parameters)

        def global_stiffness(x: np.ndarray) -> np.ndarray:
            force = np.zeros_like(x)

            for i in range(n_segments):
                segment_func = self._calculate_segment_stiffness_function(i)
                # Map global states to segment states
                segment_x = self._get_segment_states(x, i)
                segment_force = segment_func(segment_x)
                # Map segment forces back to global
                self._add_segment_forces(force, segment_force, i)

            return force

        self.stiffness_func = global_stiffness

    def create_mass_matrix(self) -> None:
        """Create global mass matrix."""
        n_segments = len(self.parameters)
        matrix_size = 6 * (n_segments + 1)

        # Lists for sparse matrix construction
        rows, cols, data = [], [], []

        for i in range(n_segments):
            m_local = self._calculate_segment_mass(i)

            # Add local matrix entries to global matrix
            for local_i in range(6):
                for local_j in range(6):
                    global_i = 6 * i + local_i
                    global_j = 6 * i + local_j
                    rows.append(global_i)
                    cols.append(global_j)
                    data.append(m_local[local_i, local_j])

        self.M = sparse.csr_matrix(
            (data, (rows, cols)), shape=(matrix_size, matrix_size)
        )

    def get_stiffness_function(self) -> Callable:
        """
        Return global stiffness function.

        Returns:
            Callable that computes global stiffness given state vector

        Raises:
            RuntimeError: If stiffness function hasn't been created
        """
        if self.stiffness_func is None:
            raise RuntimeError("Stiffness function not yet created")
        return self.stiffness_func

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

    def get_length(self) -> float:
        """
        Return total length of beam.

        Returns:
            float: Total beam length
        """
        return self.parameters["length"].sum()

    def get_segment_stiffness(self, i: int) -> Callable:
        """
        Return stiffness function for specified segment.

        Args:
            i: Segment index

        Returns:
            Callable: Segment stiffness function

        Raises:
            IndexError: If segment index is invalid
        """
        if i < 0 or i >= len(self.parameters):
            raise IndexError(f"Segment index {i} out of range")
        return self._calculate_segment_stiffness_function(i)

    def get_segment_mass(self, i: int) -> np.ndarray:
        """
        Return mass matrix for specified segment.

        Args:
            i: Segment index

        Returns:
            np.ndarray: Segment mass matrix

        Raises:
            IndexError: If segment index is invalid
        """
        if i < 0 or i >= len(self.parameters):
            raise IndexError(f"Segment index {i} out of range")
        return self._calculate_segment_mass(i)

    def _get_segment_states(self, x: np.ndarray, i: int) -> np.ndarray:
        """Extract states for segment i from global state vector."""
        start_idx = 6 * i
        return x[start_idx : start_idx + 6]

    def _add_segment_forces(
        self, global_f: np.ndarray, segment_f: np.ndarray, i: int
    ) -> None:
        """Add segment forces to global force vector."""
        start_idx = 6 * i
        global_f[start_idx : start_idx + 6] += segment_f

    def apply_boundary_conditions(
        self, conditions: Dict[int, BoundaryConditionType]
    ) -> None:
        """
        Apply multiple boundary conditions.

        Args:
            conditions: Dictionary mapping node indices to boundary condition types

        Raises:
            ValueError: If any node index is invalid
            RuntimeError: If stiffness function hasn't been created
        """
        if self.stiffness_func is None:
            raise RuntimeError(
                "Stiffness function must be created before applying boundary conditions"
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
            base_idx = node_idx * 3  # Each node has 3 components [u, θ, w]

            if bc_type == BoundaryConditionType.FIXED:
                # Constrain all DOFs for node
                dofs_to_constrain.add(base_idx)  # u (axial displacement)
                dofs_to_constrain.add(base_idx + 1)  # θ (rotation)
                dofs_to_constrain.add(base_idx + 2)  # w (transverse displacement)
            elif bc_type == BoundaryConditionType.PINNED:
                # Constrain only displacements
                dofs_to_constrain.add(base_idx)  # u (axial displacement)
                dofs_to_constrain.add(base_idx + 2)  # w (transverse displacement)
            else:
                raise ValueError(f"Unsupported boundary condition type: {bc_type}")

            self._boundary_conditions[node_idx] = bc_type

        # Create wrapper around current stiffness function
        original_stiffness = self.stiffness_func

        def stiffness_with_boundary(x: np.ndarray) -> np.ndarray:
            forces = original_stiffness(x)
            # Zero out forces for constrained DOFs
            for dof in dofs_to_constrain:
                forces[dof] = 0.0
            return forces

        # Update tracking variables
        self._constrained_dofs = dofs_to_constrain
        self._boundary_conditions_applied = True
        self.stiffness_func = stiffness_with_boundary

    def clear_boundary_conditions(self) -> None:
        """Clear all boundary conditions and recreate original stiffness function."""
        if self.stiffness_func is None:
            raise RuntimeError(
                "Stiffness function must be created before clearing boundary conditions"
            )

        # Recreate original stiffness function
        self.create_stiffness_function()

        # Clear stored boundary conditions
        self._boundary_conditions.clear()
        self._constrained_dofs.clear()
        self._boundary_conditions_applied = False

    def get_boundary_conditions(self) -> Dict[int, BoundaryConditionType]:
        """Get currently applied boundary conditions."""
        return self._boundary_conditions.copy()

    def get_constrained_dofs(self) -> Set[int]:
        """Get indices of constrained degrees of freedom."""
        return self._constrained_dofs.copy()

    def has_boundary_conditions(self) -> bool:
        """Check if boundary conditions have been applied."""
        return self._boundary_conditions_applied
