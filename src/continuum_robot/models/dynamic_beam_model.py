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
        """Precompute fluid dynamics coefficients for efficiency."""
        if self.linear_model:
            # Process linear model fluid coefficients
            wetted_areas = self.linear_params["wetted_area"].values
            drag_coefs = self.linear_params["drag_coef"].values

            # Add coefficient for final node
            wetted_areas = np.append(wetted_areas, wetted_areas[-1])
            drag_coefs = np.append(drag_coefs, drag_coefs[-1])

            # Filter out constrained DOFs
            if self.linear_constrained_dofs:
                dofs_to_keep = sorted(
                    list(
                        set(range(len(wetted_areas) * 2))
                        - set(self.linear_constrained_dofs)
                    )
                )
                node_indices = [
                    i // 2 for i in dofs_to_keep if i % 2 == 0
                ]  # Only translational DOFs

                node_areas = np.array([wetted_areas[i] for i in node_indices])
                node_drag = np.array([drag_coefs[i] for i in node_indices])

                # Precompute the drag factor for each node
                drag_factors = (
                    0.5 * self.fluid_params.fluid_density * node_drag * node_areas
                )

                # Map from velocity index to drag factor
                vel_to_drag = {
                    dofs_to_keep[i + len(node_indices)]: drag_factors[i]
                    for i in range(len(node_indices))
                }

                self.fluid_coefficients = {
                    "linear": {
                        "drag_factors": drag_factors,
                        "vel_to_drag": vel_to_drag,
                        "translation_indices": [
                            i
                            for i in range(len(dofs_to_keep) // 2)
                            if dofs_to_keep[i] % 2 == 0
                        ],
                    }
                }

        elif self.nonlinear_model:
            # Process nonlinear model fluid coefficients
            wetted_areas = self.nonlinear_params["wetted_area"].values
            drag_coefs = self.nonlinear_params["drag_coef"].values

            # Add coefficient for final node
            wetted_areas = np.append(wetted_areas, wetted_areas[-1])
            drag_coefs = np.append(drag_coefs, drag_coefs[-1])

            # Filter out constrained DOFs
            if self.nonlinear_constrained_dofs:
                dofs_to_keep = sorted(
                    list(
                        set(range(len(wetted_areas) * 3))
                        - set(self.nonlinear_constrained_dofs)
                    )
                )
                node_indices = [
                    i // 3 for i in dofs_to_keep if i % 3 == 0
                ]  # Only axial DOFs

                node_areas = np.array([wetted_areas[i] for i in node_indices])
                node_drag = np.array([drag_coefs[i] for i in node_indices])

                # Precompute the drag factor for each node
                drag_factors = (
                    0.5 * self.fluid_params.fluid_density * node_drag * node_areas
                )

                # Map from velocity index to drag factor
                n_positions = len(dofs_to_keep)
                vel_to_drag = {
                    dofs_to_keep[i] + n_positions: drag_factors[i // 3]
                    for i in range(n_positions)
                    if i % 3 == 0
                }

                self.fluid_coefficients = {
                    "nonlinear": {
                        "drag_factors": drag_factors,
                        "vel_to_drag": vel_to_drag,
                        "axial_indices": [
                            i for i in range(n_positions) if dofs_to_keep[i] % 3 == 0
                        ],
                    }
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
            # Get dimensions
            n = self.linear_model.M.shape[0]

            # Create base state space matrices
            A = sparse.bmat(
                [
                    [None, sparse.eye(n)],
                    [
                        -self.linear_M_inv.dot(self.linear_model.K),
                        -self.linear_M_inv.dot(self.linear_model.C),
                    ],
                ],
                format="csr",
            )

            if self.fluid_params.enable_fluid_effects and self.fluid_coefficients:
                # Get precomputed coefficients
                fluid_coefs = self.fluid_coefficients["linear"]
                vel_to_drag = fluid_coefs["vel_to_drag"]

                def system_with_fluid(x):
                    # Extract velocities
                    velocities = x[n:]

                    # Create drag forces vector (zeros initially)
                    drag_forces = np.zeros_like(velocities)

                    # Apply drag to translational DOFs only
                    for vel_idx, drag_factor in vel_to_drag.items():
                        vel = velocities[
                            vel_idx - n
                        ]  # Adjust index to velocities vector
                        drag_forces[vel_idx - n] = -drag_factor * vel * np.abs(vel)

                    # Apply base linear system
                    result = A.dot(x)

                    # Add drag forces to acceleration terms
                    result[n:] += self.linear_M_inv.dot(drag_forces)

                    return result

                self.system_func = system_with_fluid
            else:
                self.system_func = lambda x: A.dot(x)

        elif self.nonlinear_model:
            if self.fluid_params.enable_fluid_effects and self.fluid_coefficients:
                # Get precomputed coefficients
                fluid_coefs = self.fluid_coefficients["nonlinear"]
                vel_to_drag = fluid_coefs["vel_to_drag"]

                def nonlinear_system_with_fluid(x):
                    n = len(x) // 2
                    positions = x[:n]
                    velocities = x[n:]

                    # Calculate nonlinear stiffness forces
                    k_x = self.nonlinear_model.get_stiffness_function()(positions)

                    # Create drag forces vector (zeros initially)
                    drag_forces = np.zeros_like(velocities)

                    # Apply drag to axial DOFs only
                    for vel_idx, drag_factor in vel_to_drag.items():
                        vel = velocities[
                            vel_idx - n
                        ]  # Adjust index to velocities vector
                        drag_forces[vel_idx - n] = -drag_factor * vel * np.abs(vel)

                    # Combine position derivatives (velocities) with
                    # velocity derivatives (accelerations = forces/mass)
                    return np.concatenate(
                        [
                            velocities,
                            -self.nonlinear_M_inv.dot(k_x)
                            + self.nonlinear_M_inv.dot(drag_forces),
                        ]
                    )

                self.system_func = nonlinear_system_with_fluid
            else:

                def nonlinear_system(x: np.ndarray) -> np.ndarray:
                    n = len(x) // 2
                    positions = x[:n]
                    velocities = x[n:]

                    # Calculate stiffness forces (remove debug print)
                    k_x = self.nonlinear_model.get_stiffness_function()(positions)

                    return np.concatenate([velocities, -self.nonlinear_M_inv.dot(k_x)])

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
