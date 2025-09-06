import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile
import os
from multiprocessing import Pool, cpu_count
import time
from typing import Tuple, Any, Optional
from dataclasses import dataclass

from continuum_robot.models.dynamic_beam_model import (
    DynamicEulerBernoulliBeam,
    FluidDynamicsParams,
)


# Simulation parameters (same as original)
T_FINAL = 1  # seconds
DT = 0.001  # Time step for animation
N_SEGMENTS = 6  # Number of beam segments


@dataclass
class SimulationTask:
    """Container for a single simulation task."""

    name: str
    linear_file: str
    nonlinear_file: str
    fluid_params: Optional[FluidDynamicsParams]
    is_linear: bool


def create_beam_parameters() -> Tuple[str, str]:
    """
    Create CSV files with beam parameters for both linear and nonlinear models.

    Returns:
        tuple: (linear_file_path, nonlinear_file_path)
    """
    # Create linear beam file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write(
            "length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition,wetted_area,drag_coef\n"
        )

        # Nitinol parameters
        length = 0.25
        E = 75e9
        r = 0.005
        MInertia = np.pi * r**4 / 4
        rho = 6450
        A = np.pi * r**2
        wetted_area = 2 * np.pi * r * length
        drag_coef = 0.82

        params = [
            (length, E, MInertia, rho, A, "linear", bc, wetted_area, drag_coef)
            for bc in ["FIXED"] + ["NONE"] * (N_SEGMENTS - 1)
        ]

        for p in params:
            f.write(f"{','.join(str(x) for x in p)}\n")

        linear_file = f.name

    # Create nonlinear beam file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write(
            "length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition,wetted_area,drag_coef\n"
        )

        params = [
            (length, E, MInertia, rho, A, "nonlinear", bc, wetted_area, drag_coef)
            for bc in ["FIXED"] + ["NONE"] * (N_SEGMENTS - 1)
        ]

        for p in params:
            f.write(f"{','.join(str(x) for x in p)}\n")

        nonlinear_file = f.name

    return linear_file, nonlinear_file


def simulate_single_beam(task: SimulationTask) -> Tuple[str, Any, float]:
    """
    Simulate a single beam configuration.
    This function runs in a separate process.

    Args:
        task: Simulation task containing beam configuration

    Returns:
        Tuple of (task_name, solution, computation_time)
    """
    start_time = time.time()

    # Select appropriate file based on beam type
    param_file = task.linear_file if task.is_linear else task.nonlinear_file

    # Initialize beam model
    beam = DynamicEulerBernoulliBeam(param_file, fluid_params=task.fluid_params)
    beam.create_system_func()
    beam.create_input_func()

    # Set up simulation parameters
    t_span = (0, T_FINAL)

    # Get initial conditions
    if beam.linear_model is not None:
        n_states = beam.linear_model.M.shape[0]
    else:
        n_states = beam.nonlinear_model.M.shape[0]

    x0 = np.zeros(2 * n_states)

    # Define input force
    def u(t):
        u_vec = np.zeros(n_states)
        if t < 0.01:
            u_vec[-2] = 0.1  # Impulse at tip
        return u_vec

    # Solve system
    solution = solve_ivp(
        lambda t, x: beam.get_dynamic_system()(t, x, u(t)),
        t_span,
        x0,
        method="LSODA",
        t_eval=np.arange(t_span[0], t_span[1], DT),
    )

    computation_time = time.time() - start_time

    return task.name, solution, computation_time


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
        if linear:
            # For linear beam with 3 DOFs per node (u, w, phi)
            # State vector layout: [u1, w1, phi1, u2, w2, phi2, ..., du1_dt, dw1_dt, dphi1_dt, ...]
            # We want w components which are at indices 1, 4, 7, 10, ... (1 + 3*j)
            pos = sol.y[n_pos + 1 :: 3, i]  # Extract w (transverse) displacements
        else:
            # For nonlinear beam with 3 DOFs per node (u, w, phi)
            # Same indexing as linear now
            pos = sol.y[n_pos + 1 :: 3, i]  # Extract w (transverse) displacements

        x[i, 0] = 0  # Fixed base
        y[i, 0] = 0

        for j in range(n_segments):
            x[i, j + 1] = x[i, j] + dx
            y[i, j + 1] = pos[j] if j < len(pos) else 0

    return x, y


def main():
    """Main function to run parallel beam comparison."""

    print("=" * 60)
    print("PARALLEL BEAM COMPARISON")
    print("=" * 60)
    print(f"Running on {cpu_count()} CPU cores")
    print(f"Simulating {N_SEGMENTS} segments for {T_FINAL} seconds")
    print("-" * 60)

    # Create parameter files
    linear_file, nonlinear_file = create_beam_parameters()

    try:
        # Create fluid dynamics parameters
        fluid_params = FluidDynamicsParams(
            fluid_density=1000.0, enable_fluid_effects=True
        )

        # Define all simulation tasks
        tasks = [
            SimulationTask(
                "Linear (No Fluid)", linear_file, nonlinear_file, None, True
            ),
            SimulationTask(
                "Linear (Fluid)", linear_file, nonlinear_file, fluid_params, True
            ),
            SimulationTask(
                "Nonlinear (No Fluid)", linear_file, nonlinear_file, None, False
            ),
            SimulationTask(
                "Nonlinear (Fluid)", linear_file, nonlinear_file, fluid_params, False
            ),
        ]

        # Run simulations in parallel
        print("Starting parallel simulations...")
        overall_start = time.time()

        with Pool(processes=min(4, cpu_count())) as pool:
            results = pool.map(simulate_single_beam, tasks)

        overall_time = time.time() - overall_start

        # Process results
        solutions = {}
        computation_times = {}

        for name, sol, comp_time in results:
            solutions[name] = sol
            computation_times[name] = comp_time
            print(f"  {name}: {comp_time:.3f}s")

        total_sequential_time = sum(computation_times.values())
        speedup = total_sequential_time / overall_time

        print("-" * 60)
        print(f"Total parallel time: {overall_time:.3f}s")
        print(f"Sequential time would be: {total_sequential_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        print("=" * 60)

        # Extract beam shapes
        print("\nExtracting beam shapes...")
        shapes = {}

        # Determine dx based on model type
        dx_lin = 1.5 / N_SEGMENTS  # Total length / segments
        dx_nonlin = 1.5 / N_SEGMENTS

        for name, sol in solutions.items():
            is_linear = "Linear" in name
            dx = dx_lin if is_linear else dx_nonlin
            x, y = extract_beam_shapes(sol, N_SEGMENTS, dx, linear=is_linear)
            shapes[name] = (x, y)

        # Create visualization
        print("Creating visualization...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Animation plot setup
        ax1.set_xlim(0.0, 1.6)

        # Find global y-limits
        y_values = np.concatenate([y.flatten() for _, y in shapes.values()])
        y_min, y_max = np.min(y_values), np.max(y_values)
        y_pad = (y_max - y_min) * 0.1
        ax1.set_ylim(y_min - y_pad, y_max + y_pad)

        ax1.set_xlabel("Beam Length (m)")
        ax1.set_ylabel("Displacement (m)")
        ax1.set_title("Beam Response Comparison (Parallel Computation)")
        ax1.grid(True)

        # Create lines for animation
        lines = {}
        colors = {"Linear": "b", "Nonlinear": "r"}
        styles = {"No Fluid": "-", "Fluid": "--"}

        for name, (x, y) in shapes.items():
            beam_type = "Linear" if "Linear" in name else "Nonlinear"
            fluid_type = "Fluid" if "Fluid" in name else "No Fluid"

            color = colors[beam_type]
            style = styles[fluid_type]

            (line,) = ax1.plot(
                [], [], color=color, linestyle=style, linewidth=2, label=name
            )
            lines[name] = line

        ax1.legend()

        # Tip displacement plot
        for name, (x, y) in shapes.items():
            beam_type = "Linear" if "Linear" in name else "Nonlinear"
            fluid_type = "Fluid" if "Fluid" in name else "No Fluid"

            color = colors[beam_type]
            style = styles[fluid_type]

            ax2.plot(
                solutions[name].t,
                y[:, -1],
                color=color,
                linestyle=style,
                linewidth=2,
                label=name,
            )

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Tip Displacement (m)")
        ax2.set_title("Beam Tip Response")
        ax2.grid(True)
        ax2.legend()

        # Animation function
        def animate(frame):
            for name, (x, y) in shapes.items():
                if frame < len(x):
                    lines[name].set_data(x[frame], y[frame])
            return list(lines.values())

        # Create animation
        n_frames = min(len(x) for x, _ in shapes.values())
        anim = FuncAnimation(
            fig=fig,
            func=animate,
            frames=n_frames,
            interval=DT * 1000,  # Convert to milliseconds
            blit=True,
        )

        plt.tight_layout()
        plt.show()

        return anim  # Return to prevent garbage collection

    finally:
        # Cleanup temporary files
        os.unlink(linear_file)
        os.unlink(nonlinear_file)
        print("\nTemporary files cleaned up")


if __name__ == "__main__":
    anim = main()  # Store animation to prevent garbage collection
