"""
Utilities for beam comparison examples.

This module contains shared functionality for different beam comparison scripts,
including simulation parameters, material properties, helper functions, and
common plotting utilities.
"""

import numpy as np
import tempfile
import os
import time
from typing import Tuple, Any, Optional
from dataclasses import dataclass

from continuum_robot.models.dynamic_beam_model import DynamicEulerBernoulliBeam
from continuum_robot.models.force_params import ForceParams

# Simulation parameters
T_FINAL = 0.1  # seconds (reduced for faster testing)
DT = 0.001  # Time step for animation
N_SEGMENTS = 6  # Number of beam segments

# Material properties (Nitinol)
MATERIAL_PROPS = {"length": 0.25, "E": 75e9, "r": 0.005, "rho": 6450, "drag_coef": 0.82}


def get_material_properties():
    """Calculate derived material properties from base parameters."""
    props = MATERIAL_PROPS.copy()
    props["MInertia"] = np.pi * props["r"] ** 4 / 4
    props["A"] = np.pi * props["r"] ** 2
    props["wetted_area"] = 2 * np.pi * props["r"] * props["length"]
    return props


def create_csv_file(beam_types, boundary_conditions):
    """
    Create a CSV file with beam parameters.

    Args:
        beam_types: List of beam types for each segment
        boundary_conditions: List of boundary conditions for each segment

    Returns:
        str: Path to created temporary file
    """
    props = get_material_properties()

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write(
            "length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition,wetted_area,drag_coef\n"
        )

        params = [
            (
                props["length"],
                props["E"],
                props["MInertia"],
                props["rho"],
                props["A"],
                beam_type,
                bc,
                props["wetted_area"],
                props["drag_coef"],
            )
            for beam_type, bc in zip(beam_types, boundary_conditions)
        ]

        for p in params:
            f.write(f"{','.join(str(x) for x in p)}\n")

        return f.name


@dataclass
class SimulationTask:
    """Container for a single simulation task."""

    name: str
    param_file: str
    force_params: Optional[ForceParams] = None

    def __post_init__(self):
        """Set default force params if none provided."""
        if self.force_params is None:
            self.force_params = ForceParams()


def create_beam_parameters() -> Tuple[str, str, str]:
    """
    Create CSV files with beam parameters for linear, nonlinear, and mixed models.

    Returns:
        tuple: (linear_file_path, nonlinear_file_path, mixed_file_path)
    """
    boundary_conditions = ["FIXED"] + ["NONE"] * (N_SEGMENTS - 1)

    # Create linear beam file
    linear_types = ["linear"] * N_SEGMENTS
    linear_file = create_csv_file(linear_types, boundary_conditions)

    # Create nonlinear beam file
    nonlinear_types = ["nonlinear"] * N_SEGMENTS
    nonlinear_file = create_csv_file(nonlinear_types, boundary_conditions)

    # Create mixed beam file: Linear base segments, nonlinear tip segments
    mixed_types = ["linear"] * (N_SEGMENTS // 2) + ["nonlinear"] * (
        N_SEGMENTS - N_SEGMENTS // 2
    )
    mixed_file = create_csv_file(mixed_types, boundary_conditions)

    return linear_file, nonlinear_file, mixed_file


def simulate_single_beam(task: SimulationTask) -> Tuple[str, Any, float, dict]:
    """
    Simulate a single beam configuration.
    This function can run in a separate process.

    Args:
        task: Simulation task containing beam configuration

    Returns:
        Tuple of (task_name, solution, computation_time, solver_stats)
    """
    start_time = time.time()

    # Initialize beam model with the provided parameter file
    beam = DynamicEulerBernoulliBeam(task.param_file, force_params=task.force_params)

    beam.create_system_func()
    beam.create_input_func()

    # Set up simulation parameters
    t_span = (0, T_FINAL)

    # Get initial conditions from unified beam model
    n_states = beam.beam_model.M.shape[0]

    x0 = np.zeros(2 * n_states)

    # Define input force
    def u(t):
        u_vec = np.zeros(n_states)
        if t < 0.01:
            u_vec[-2] = 0.1  # Impulse at tip
        return u_vec

    # Solve system
    from scipy.integrate import solve_ivp

    solution = solve_ivp(
        lambda t, x: beam.get_dynamic_system()(t, x, u(t)),
        t_span,
        x0,
        method="LSODA",
        t_eval=np.arange(t_span[0], t_span[1], DT),
    )

    computation_time = time.time() - start_time

    # Extract solver statistics
    solver_stats = {
        "nfev": solution.nfev if hasattr(solution, "nfev") else 0,
        "njev": solution.njev if hasattr(solution, "njev") else 0,
        "nlu": solution.nlu if hasattr(solution, "nlu") else 0,
    }

    return task.name, solution, computation_time, solver_stats


def extract_beam_shapes(
    sol: Any, n_segments: int, dx: float, linear: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract beam x,y coordinates over time.

    Args:
        sol: Solution from solve_ivp
        n_segments: Number of beam segments
        dx: Segment length
        linear: Whether this is a linear beam

    Returns:
        Tuple of (x_coordinates, y_coordinates)
    """
    n_pos = len(sol.y) // 2
    n_points = n_segments + 1

    x = np.zeros((len(sol.t), n_points))
    y = np.zeros((len(sol.t), n_points))

    for i in range(len(sol.t)):
        # For beams with 3 DOFs per node (u, w, phi)
        pos = sol.y[n_pos + 1 :: 3, i]  # Extract w (transverse) displacements

        x[i, 0] = 0  # Fixed base
        y[i, 0] = 0

        for j in range(n_segments):
            x[i, j + 1] = x[i, j] + dx
            y[i, j + 1] = pos[j] if j < len(pos) else 0

    return x, y


def calculate_natural_frequencies(
    length, elastic_modulus, moment_inertia, density, cross_area
):
    """
    Calculate the first 4 natural frequencies of a cantilever beam using Euler-Bernoulli theory.

    Args:
        length: Beam length (m)
        elastic_modulus: Young's modulus (Pa)
        moment_inertia: Second moment of area (m^4)
        density: Material density (kg/m^3)
        cross_area: Cross-sectional area (m^2)

    Returns:
        List of first 4 natural frequencies (Hz)
    """
    # Eigenvalue constants for cantilever beam (beta_n * L)
    beta_L = [0.596864 * np.pi, 1.49418 * np.pi, 2.50025 * np.pi, 3.49999 * np.pi]

    # Natural frequency formula for cantilever beam
    # f_n = (beta_n * L)^2 * sqrt(EI / (rho * A * L^4)) / (2 * pi)
    frequencies = []
    for bl in beta_L:
        freq = (
            (bl**2)
            * np.sqrt(
                elastic_modulus * moment_inertia / (density * cross_area * length**4)
            )
            / (2 * np.pi)
        )
        frequencies.append(freq)

    return frequencies


def get_beam_type_and_style(name: str, force_styles: dict = None):
    """
    Extract beam type and force conditions from task name and return styling info.

    Args:
        name: Task name (e.g., "Linear (No Fluid)", "Linear (Gravity)")
        force_styles: Dictionary mapping force types to line styles

    Returns:
        tuple: (beam_type, force_type, color, style)
    """
    # Define colors and default styles
    colors = {"Linear": "b", "Nonlinear": "r", "Mixed Lin-Base/Nonlin-Tip": "g"}
    default_styles = {"No Fluid": "-", "Fluid": "--", "Gravity": ":"}

    # Use provided force_styles or default
    styles = force_styles if force_styles is not None else default_styles

    # Determine beam type from name
    if "Linear" in name and "Mixed" not in name:
        beam_type = "Linear"
    elif "Nonlinear" in name and "Mixed" not in name:
        beam_type = "Nonlinear"
    elif "Lin-Base/Nonlin-Tip" in name:
        beam_type = "Mixed Lin-Base/Nonlin-Tip"
    else:
        beam_type = "Linear"  # fallback

    # Determine force type (priority: Gravity > Fluid > No Fluid)
    if "(Gravity)" in name:
        force_type = "Gravity"
    elif "(Fluid)" in name:
        force_type = "Fluid"
    else:
        force_type = "No Fluid"

    # Get color and style
    color = colors.get(beam_type, "black")
    style = styles.get(force_type, "-")

    return beam_type, force_type, color, style


def cleanup_temp_files(*file_paths):
    """
    Clean up temporary files.

    Args:
        *file_paths: Variable number of file paths to clean up
    """
    for temp_file in file_paths:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


def print_performance_table(computation_times, solver_statistics):
    """
    Print a formatted performance table.

    Args:
        computation_times: Dictionary of task names to computation times
        solver_statistics: Dictionary of task names to solver stats
    """
    print("\nDETAILED SOLVER PERFORMANCE:")
    print("=" * 120)

    # Table header
    header = f"{'Configuration':<35} {'Time (s)':<10} {'nfev':<8} {'njev':<8} {'nlu':<6} {'nfev/s':<10} {'Efficiency':<12}"
    print(header)
    print("-" * 120)

    # Table rows
    for name in computation_times.keys():
        time_val = computation_times[name]
        stats = solver_statistics[name]
        nfev = stats["nfev"]
        njev = stats["njev"]
        nlu = stats["nlu"]
        nfev_per_sec = nfev / time_val if time_val > 0 else 0

        # Calculate efficiency metric (lower is better for time, higher for nfev/s)
        efficiency = f"{nfev_per_sec:.0f} eval/s"

        row = f"{name:<35} {time_val:<10.3f} {nfev:<8} {njev:<8} {nlu:<6} {nfev_per_sec:<10.0f} {efficiency:<12}"
        print(row)

    print("-" * 120)
