"""
Parallel beam comparison focused on fluid dynamics effects.

This example compares the behavior of linear, nonlinear, and mixed beam models
with and without fluid dynamics effects, running simulations in parallel for
better performance.
"""

import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool, cpu_count
import time

from example_utilities import (
    SimulationTask,
    create_beam_parameters,
    simulate_single_beam,
    extract_beam_shapes,
    calculate_natural_frequencies,
    get_beam_type_and_style,
    cleanup_temp_files,
    print_performance_table,
    get_material_properties,
    T_FINAL,
    DT,
    N_SEGMENTS,
)


def main():
    """Main function to run parallel fluid dynamics beam comparison."""

    print("=" * 60)
    print("PARALLEL FLUID DYNAMICS BEAM COMPARISON")
    print("=" * 60)
    print(f"Running on {cpu_count()} CPU cores")
    print(f"Simulating {N_SEGMENTS} segments for {T_FINAL} seconds")
    print("-" * 60)

    # Create parameter files
    linear_file, nonlinear_file, mixed_file = create_beam_parameters()

    try:
        # Create force parameters
        from continuum_robot.models.force_params import ForceParams

        no_forces_params = ForceParams()  # Default: no forces enabled
        fluid_params = ForceParams(fluid_density=1000.0, enable_fluid_effects=True)

        # Define simulation tasks focused on fluid effects
        tasks = [
            # No fluid forces
            SimulationTask(
                "Linear (No Fluid)", linear_file, force_params=no_forces_params
            ),
            SimulationTask(
                "Nonlinear (No Fluid)", nonlinear_file, force_params=no_forces_params
            ),
            SimulationTask(
                "Mixed Lin-Base/Nonlin-Tip (No Fluid)",
                mixed_file,
                force_params=no_forces_params,
            ),
            # With fluid forces
            SimulationTask("Linear (Fluid)", linear_file, force_params=fluid_params),
            SimulationTask(
                "Nonlinear (Fluid)", nonlinear_file, force_params=fluid_params
            ),
            SimulationTask(
                "Mixed Lin-Base/Nonlin-Tip (Fluid)",
                mixed_file,
                force_params=fluid_params,
            ),
        ]

        # Run simulations in parallel
        print("Starting parallel simulations...")
        overall_start = time.time()

        with Pool(processes=min(len(tasks), cpu_count())) as pool:
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
        print_performance_table(computation_times, solver_statistics)

        # Extract beam shapes
        print("\nExtracting beam shapes...")
        shapes = {}

        # All models use the same segment length in unified architecture
        dx = 1.5 / N_SEGMENTS  # Total beam length / segments

        for name, sol in solutions.items():
            x, y = extract_beam_shapes(sol, N_SEGMENTS, dx, linear=False)
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
        ax1.set_title("Fluid Dynamics Effects on Beam Response (Parallel Computation)")
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
        ax2.set_title("Beam Tip Response - Fluid Effects Comparison")
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
        ax3.set_title("FFT Analysis - Fluid Damping Effects")
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
        cleanup_temp_files(linear_file, nonlinear_file, mixed_file)
        print("\nTemporary files cleaned up")


if __name__ == "__main__":
    anim = main()  # Store animation to prevent garbage collection
