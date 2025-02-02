from typing import Callable, Dict, Union
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import inv
import pathlib
from enum import Enum

from models.linear_euler_bernoulli_beam import (
    LinearEulerBernoulliBeam,
    BoundaryConditionType,
)
from models.nonlinear_euler_bernoulli_beam import NonlinearEulerBernoulliBeam


class ElementType(Enum):
    """Enumeration of supported element types."""

    LINEAR = "linear"
    NONLINEAR = "nonlinear"


class DynamicEulerBernoulliBeam:
    """
    Dynamic model combining linear and nonlinear Euler-Bernoulli beam elements.

    This class creates a dynamic system that can be used with scipy.integrate.solve_ivp.
    The system handles both linear and nonlinear elements defined by their type in the
    input CSV file.
    """

    def __init__(self, filename: Union[str, pathlib.Path], damping_ratio: float = 0.01):
        """
        Initialize dynamic beam model from CSV file.

        Args:
            filename: Path to CSV containing beam parameters and element types
            damping_ratio: Damping ratio for linear elements [0,1]

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If parameters are invalid
        """
        # Read and validate parameters
        self.params = pd.read_csv(filename)
        self._validate_parameters()

        # Split elements by type
        linear_mask = self.params["type"].str.lower() == ElementType.LINEAR.value
        self.linear_params = self.params[linear_mask].copy()
        self.nonlinear_params = self.params[~linear_mask].copy()

        # Create boundary conditions dictionary
        self.boundary_conditions = self._process_boundary_conditions()

        # Initialize beam models if elements exist
        self.linear_model = None
        if not self.linear_params.empty:
            self.linear_model = LinearEulerBernoulliBeam(
                self.linear_params, damping_ratio
            )
            self.linear_model.apply_boundary_conditions(self.boundary_conditions)

        self.nonlinear_model = None
        if not self.nonlinear_params.empty:
            self.nonlinear_model = NonlinearEulerBernoulliBeam(self.nonlinear_params)
            self.nonlinear_model.create_mass_matrix()
            self.nonlinear_model.create_stiffness_function()
            self.nonlinear_model.apply_boundary_conditions(self.boundary_conditions)

        # Initialize system functions
        self.system_func = None
        self.input_func = None

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

    def _process_boundary_conditions(self) -> Dict[int, BoundaryConditionType]:
        """Process boundary conditions from parameters."""
        conditions = {}
        for i, bc in enumerate(self.params["boundary_condition"]):
            if bc == "FIXED":
                conditions[i] = BoundaryConditionType.FIXED
            elif bc == "PINNED":
                conditions[i] = BoundaryConditionType.PINNED
        return conditions

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
            M_inv = inv(self.linear_model.M)
            A = sparse.bmat(
                [
                    [None, sparse.eye(n)],
                    [-M_inv.dot(self.linear_model.K), -M_inv.dot(self.linear_model.C)],
                ]
            )
            self.system_func = lambda x: A.dot(x)

        elif self.nonlinear_model:
            # Create nonlinear system
            M_inv = inv(self.nonlinear_model.M)

            def nonlinear_system(x: np.ndarray) -> np.ndarray:
                n = len(x) // 2
                positions = x[:n]
                velocities = x[n:]

                print(positions)
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
                M_inv = inv(self.linear_model.M)
            else:
                M_inv = inv(self.nonlinear_model.M)

            B = sparse.bmat([[sparse.csr_matrix((n, n))], [M_inv]])
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
