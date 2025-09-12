import numpy as np
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
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

# Material properties (Nitinol)
MATERIAL_PROPS = {"length": 0.25, "E": 75e9, "r": 0.005, "rho": 6450, "drag_coef": 0.82}


# Derived properties
def get_material_properties():
    """Calculate derived material properties from base parameters."""
    props = MATERIAL_PROPS.copy()
    props["MInertia"] = np.pi * props["r"] ** 4 / 4
    props["A"] = np.pi * props["r"] ** 2
    props["wetted_area"] = 2 * np.pi * props["r"] * props["length"]
    return props


def get_beam_type_and_style(name: str):
    """
    Extract beam type and fluid condition from task name and return styling info.

    Args:
        name: Task name (e.g., "Linear (No Fluid)")

    Returns:
        tuple: (beam_type, fluid_type, color, style)
    """
    # Define colors and styles
    colors = {"Linear": "b", "Nonlinear": "r", "Mixed Lin-Base/Nonlin-Tip": "g"}
    styles = {"No Fluid": "-", "Fluid": "--"}

    # Determine beam type from name
    if "Linear" in name and "Mixed" not in name:
        beam_type = "Linear"
    elif "Nonlinear" in name and "Mixed" not in name:
        beam_type = "Nonlinear"
    elif "Lin-Base/Nonlin-Tip" in name:
        beam_type = "Mixed Lin-Base/Nonlin-Tip"
    else:
        beam_type = "Linear"  # fallback

    # Determine fluid type
    fluid_type = "Fluid" if "(Fluid)" in name else "No Fluid"

    # Get color and style
    color = colors.get(beam_type, "black")
    style = styles[fluid_type]

    return beam_type, fluid_type, color, style


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
    fluid_params: Optional[FluidDynamicsParams]


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
    print(f"Linear beam file: {linear_file}")

    # Create nonlinear beam file
    nonlinear_types = ["nonlinear"] * N_SEGMENTS
    nonlinear_file = create_csv_file(nonlinear_types, boundary_conditions)

    # Create mixed beam file: Linear base segments, nonlinear tip segments
    mixed_types = ["linear"] * (N_SEGMENTS // 2) + ["nonlinear"] * (
        N_SEGMENTS - N_SEGMENTS // 2
    )
    mixed_file = create_csv_file(mixed_types, boundary_conditions)

    return linear_file, nonlinear_file, mixed_file


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

    # Initialize beam model with the provided parameter file
    beam = DynamicEulerBernoulliBeam(task.param_file, fluid_params=task.fluid_params)
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


def main():
    """Main function to run parallel beam comparison."""

    print("=" * 60)
    print("PARALLEL BEAM COMPARISON")
    print("=" * 60)
    print(f"Running on {cpu_count()} CPU cores")
    print(f"Simulating {N_SEGMENTS} segments for {T_FINAL} seconds")
    print("-" * 60)

    # Create parameter files
    linear_file, nonlinear_file, mixed_file = create_beam_parameters()

    try:
        # Create fluid dynamics parameters
        fluid_params = FluidDynamicsParams(
            fluid_density=1000.0, enable_fluid_effects=True
        )

        # Define all simulation tasks
        tasks = [
            SimulationTask("Linear (No Fluid)", linear_file, None),
            SimulationTask("Linear (Fluid)", linear_file, fluid_params),
            SimulationTask("Nonlinear (No Fluid)", nonlinear_file, None),
            SimulationTask("Nonlinear (Fluid)", nonlinear_file, fluid_params),
            SimulationTask("Mixed Lin-Base/Nonlin-Tip (No Fluid)", mixed_file, None),
            SimulationTask(
                "Mixed Lin-Base/Nonlin-Tip (Fluid)", mixed_file, fluid_params
            ),
        ]

        # Run simulations in parallel
        print("Starting parallel simulations...")
        overall_start = time.time()

        with Pool(processes=min(6, cpu_count())) as pool:
            results = pool.map(simulate_single_beam, tasks)

        overall_time = time.time() - overall_start

        # Process results
        solutions = {}
        computation_times = {}
        solver_statistics = {}

        for name, sol, comp_time, stats in results:
            solutions[name] = sol
            computation_times[name] = comp_time
            solver_statistics[name] = stats

        total_sequential_time = sum(computation_times.values())
        speedup = total_sequential_time / overall_time

        print("-" * 60)
        print(f"Total parallel time: {overall_time:.3f}s")
        print(f"Sequential time would be: {total_sequential_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        print("=" * 60)

        # Create detailed performance table
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

        # Extract beam shapes
        print("\nExtracting beam shapes...")
        shapes = {}

        # All models use the same segment length in unified architecture
        dx = 1.5 / N_SEGMENTS  # Total beam length / segments

        for name, sol in solutions.items():
            x, y = extract_beam_shapes(
                sol, N_SEGMENTS, dx, linear=False
            )  # Use consistent indexing
            shapes[name] = (x, y)

        # Create visualization
        print("Creating visualization...")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))

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
        for name, (x, y) in shapes.items():
            _, _, color, style = get_beam_type_and_style(name)
            (line,) = ax1.plot(
                [], [], color=color, linestyle=style, linewidth=2, label=name
            )
            lines[name] = line

        ax1.legend()

        # Tip displacement plot
        for name, (x, y) in shapes.items():
            _, _, color, style = get_beam_type_and_style(name)
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

        # FFT Analysis plot (ax3)
        # Calculate theoretical natural frequencies using beam parameters
        props = get_material_properties()
        total_length = props["length"] * N_SEGMENTS
        natural_freqs = calculate_natural_frequencies(
            total_length, props["E"], props["MInertia"], props["rho"], props["A"]
        )

        for name, (x, y) in shapes.items():
            # Extract tip displacement (last column)
            tip_displacement = y[:, -1]

            # Get time array from solution
            t = solutions[name].t
            dt = t[1] - t[0]  # Time step

            # Compute FFT
            fft_values = fft(tip_displacement)
            frequencies = fftfreq(len(tip_displacement), dt)

            # Only plot positive frequencies
            positive_freq_mask = frequencies > 0
            frequencies_pos = frequencies[positive_freq_mask]
            fft_magnitude = np.abs(fft_values[positive_freq_mask])

            # Get styling info
            _, _, color, style = get_beam_type_and_style(name)

            # Plot FFT magnitude spectrum
            ax3.semilogy(
                frequencies_pos,
                fft_magnitude,
                color=color,
                linestyle=style,
                linewidth=2,
                label=name,
            )

        # Plot theoretical natural frequencies as vertical lines
        for i, freq in enumerate(natural_freqs):
            ax3.axvline(
                x=freq,
                color="black",
                linestyle=":",
                alpha=0.7,
                label=f"Natural Freq {i+1}: {freq:.1f} Hz"
                if i == 0
                else f"{freq:.1f} Hz",
            )

        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Magnitude")
        ax3.set_title("FFT Analysis of Beam Tip Displacements")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_xlim(0, 50)  # Focus on lower frequencies

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
        for temp_file in [linear_file, nonlinear_file, mixed_file]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        print("\nTemporary files cleaned up")


if __name__ == "__main__":
    anim = main()  # Store animation to prevent garbage collection
