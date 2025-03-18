import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile
import os

from continuum_robot.models.dynamic_beam_model import (
    DynamicEulerBernoulliBeam,
    FluidDynamicsParams,
)

# Simulation parameters
T_FINAL = 1  # seconds
DT = 0.001  # Time step for animation
N_SEGMENTS = 6  # Number of beam segments


def create_beam_parameters():
    """Create CSV files with beam parameters for both linear and nonlinear models.

    Returns:
        tuple: (linear_file_path, nonlinear_file_path)
    """
    # Create linear beam file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        # Write header
        f.write(
            "length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition,wetted_area,drag_coef\n"
        )

        # Nitinol parameters
        length = 0.25  # Each segment length for 1.5m total
        E = 75e9  # Young's modulus (Pa)
        r = 0.005  # Radius (m)
        MInertia = np.pi * r**4 / 4  # Moment of inertia
        rho = 6450  # Density (kg/m³)
        A = np.pi * r**2  # Cross-sectional area
        wetted_area = (
            2 * np.pi * r * length
        )  # Wetted area (m²) - cylindrical surface area
        drag_coef = 0.82  # Drag coefficient for a long cylinder

        # Create N_SEGMENTS segments, first has fixed boundary condition
        params = [
            (length, E, MInertia, rho, A, beam_type, bc, wetted_area, drag_coef)
            for beam_type, bc in [("linear", "FIXED")]
            + [("linear", "NONE")] * (N_SEGMENTS - 1)
        ]

        for p in params:
            f.write(f"{','.join(str(x) for x in p)}\n")

        linear_file = f.name

    # Create nonlinear beam file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        # Write header
        f.write(
            "length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition,wetted_area,drag_coef\n"
        )

        # Use same parameters as linear model, just change the type
        params = [
            (length, E, MInertia, rho, A, beam_type, bc, wetted_area, drag_coef)
            for beam_type, bc in [("nonlinear", "FIXED")]
            + [("nonlinear", "NONE")] * (N_SEGMENTS - 1)
        ]

        for p in params:
            f.write(f"{','.join(str(x) for x in p)}\n")

        nonlinear_file = f.name

    return linear_file, nonlinear_file


def setup_initial_conditions(beam):
    """Set up initial conditions."""
    if hasattr(beam.linear_model, "M"):
        n_states = beam.linear_model.M.shape[0]
    else:
        n_states = beam.nonlinear_model.M.shape[0]

    # Zero initial positions and velocities except tip velocity
    x0 = np.zeros(2 * n_states)

    return x0


def solve_beam_dynamics(beam, x0, t_span):
    """Solve beam dynamics using solve_ivp."""

    # Zero input force
    def u(t):
        u = np.zeros(len(x0) // 2)
        if t < 0.01:
            u[-2] = 0.1  # Impulse at tip
        return u

    # Solve system
    sol_nonlinear = solve_ivp(
        lambda t, x: beam(t, x, u),
        t_span,
        x0,
        method="LSODA",
    )

    return sol_nonlinear


def extract_beam_shapes(sol, n_segments, dx, linear=False):
    """Extract beam x,y coordinates over time."""
    n_pos = len(sol.y) // 2
    n_points = n_segments + 1

    # Initialize arrays
    x = np.zeros((len(sol.t), n_points))
    y = np.zeros((len(sol.t), n_points))

    # For each time step
    for i in range(len(sol.t)):
        # Get positions at this time
        if linear:
            pos = sol.y[n_pos::2, i]
        else:
            pos = sol.y[n_pos + 1 :: 3, i]

        # Build beam shape
        x[i, 0] = 0  # Fixed base
        y[i, 0] = 0

        for j in range(0, n_segments):
            # Add segment projected length
            x[i, j + 1] = x[i, j] + dx
            # Add displacement
            y[i, j + 1] = pos[j]

    return x, y


def main():
    # Create parameter files
    linear_file, nonlinear_file = create_beam_parameters()

    try:
        # Initialize beam models - 4 configurations
        # Create fluid dynamics parameters
        fluid_params = FluidDynamicsParams(
            fluid_density=1000.0, enable_fluid_effects=True  # Water density (kg/m³)
        )

        # Initialize all four beam configurations
        lin_no_fluid = DynamicEulerBernoulliBeam(linear_file, fluid_params=None)
        lin_fluid = DynamicEulerBernoulliBeam(linear_file, fluid_params=fluid_params)
        nonlin_no_fluid = DynamicEulerBernoulliBeam(nonlinear_file, fluid_params=None)
        nonlin_fluid = DynamicEulerBernoulliBeam(
            nonlinear_file, fluid_params=fluid_params
        )

        # Create system functions for all beams
        for beam in [lin_no_fluid, lin_fluid, nonlin_no_fluid, nonlin_fluid]:
            beam.create_system_func()
            beam.create_input_func()

        # Set up simulation parameters
        t_span = (0, T_FINAL)

        # Get initial conditions and solve dynamics
        x0_lin_no_fluid = setup_initial_conditions(lin_no_fluid)
        sol_lin_no_fluid = solve_beam_dynamics(
            lin_no_fluid.get_dynamic_system(), x0_lin_no_fluid, t_span
        )

        x0_lin_fluid = setup_initial_conditions(lin_fluid)
        sol_lin_fluid = solve_beam_dynamics(
            lin_fluid.get_dynamic_system(), x0_lin_fluid, t_span
        )

        x0_nonlin_no_fluid = setup_initial_conditions(nonlin_no_fluid)
        sol_nonlin_no_fluid = solve_beam_dynamics(
            nonlin_no_fluid.get_dynamic_system(), x0_nonlin_no_fluid, t_span
        )

        x0_nonlin_fluid = setup_initial_conditions(nonlin_fluid)
        sol_nonlin_fluid = solve_beam_dynamics(
            nonlin_fluid.get_dynamic_system(), x0_nonlin_fluid, t_span
        )

        # Extract beam shapes
        dx_lin = lin_no_fluid.linear_model.get_length() / N_SEGMENTS
        dx_nonlin = nonlin_no_fluid.nonlinear_model.get_length() / N_SEGMENTS

        x_lin_no_fluid, y_lin_no_fluid = extract_beam_shapes(
            sol_lin_no_fluid, N_SEGMENTS, dx_lin, linear=True
        )
        x_lin_fluid, y_lin_fluid = extract_beam_shapes(
            sol_lin_fluid, N_SEGMENTS, dx_lin, linear=True
        )
        x_nonlin_no_fluid, y_nonlin_no_fluid = extract_beam_shapes(
            sol_nonlin_no_fluid, N_SEGMENTS, dx_nonlin
        )
        x_nonlin_fluid, y_nonlin_fluid = extract_beam_shapes(
            sol_nonlin_fluid, N_SEGMENTS, dx_nonlin
        )

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Animation plot
        ax1.set_xlim(0.0, 1.6)

        # Add padding to y limits - find global min/max across all simulations
        y_values = np.concatenate(
            [
                y_lin_no_fluid.flatten(),
                y_lin_fluid.flatten(),
                y_nonlin_no_fluid.flatten(),
                y_nonlin_fluid.flatten(),
            ]
        )
        y_min, y_max = np.min(y_values), np.max(y_values)
        y_pad = (y_max - y_min) * 0.1
        ax1.set_ylim(y_min - y_pad, y_max + y_pad)

        ax1.set_xlabel("Beam Length (m)")
        ax1.set_ylabel("Displacement (m)")
        ax1.set_title("Beam Response Comparison")
        ax1.grid(True)

        # Create lines for each beam type with distinct colors and styles
        (line_lin_no_fluid,) = ax1.plot(
            [], [], "b-", linewidth=2, label="Linear (No Fluid)"
        )
        (line_lin_fluid,) = ax1.plot([], [], "b--", linewidth=2, label="Linear (Fluid)")
        (line_nonlin_no_fluid,) = ax1.plot(
            [], [], "r-", linewidth=2, label="Nonlinear (No Fluid)"
        )
        (line_nonlin_fluid,) = ax1.plot(
            [], [], "r--", linewidth=2, label="Nonlinear (Fluid)"
        )

        ax1.legend()

        # Tip displacement plot
        ax2.plot(
            sol_lin_no_fluid.t,
            y_lin_no_fluid[:, -1],
            "b-",
            linewidth=2,
            label="Linear (No Fluid)",
        )
        ax2.plot(
            sol_lin_fluid.t,
            y_lin_fluid[:, -1],
            "b--",
            linewidth=2,
            label="Linear (Fluid)",
        )
        ax2.plot(
            sol_nonlin_no_fluid.t,
            y_nonlin_no_fluid[:, -1],
            "r-",
            linewidth=2,
            label="Nonlinear (No Fluid)",
        )
        ax2.plot(
            sol_nonlin_fluid.t,
            y_nonlin_fluid[:, -1],
            "r--",
            linewidth=2,
            label="Nonlinear (Fluid)",
        )

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Tip Displacement (m)")
        ax2.set_title("Beam Tip Response")
        ax2.grid(True)
        ax2.legend()

        # Create common time grid for animation
        # Use a uniform time grid and find the closest time point in each solution
        common_times = np.linspace(0, T_FINAL, int(T_FINAL / DT) + 1)

        def animate(frame):
            # Get current time
            t = common_times[frame]

            # Find closest time points in each solution
            idx_lin_no_fluid = np.abs(sol_lin_no_fluid.t - t).argmin()
            idx_lin_fluid = np.abs(sol_lin_fluid.t - t).argmin()
            idx_nonlin_no_fluid = np.abs(sol_nonlin_no_fluid.t - t).argmin()
            idx_nonlin_fluid = np.abs(sol_nonlin_fluid.t - t).argmin()

            # Update beam shapes
            line_lin_no_fluid.set_data(
                x_lin_no_fluid[idx_lin_no_fluid], y_lin_no_fluid[idx_lin_no_fluid]
            )
            line_lin_fluid.set_data(
                x_lin_fluid[idx_lin_fluid], y_lin_fluid[idx_lin_fluid]
            )
            line_nonlin_no_fluid.set_data(
                x_nonlin_no_fluid[idx_nonlin_no_fluid],
                y_nonlin_no_fluid[idx_nonlin_no_fluid],
            )
            line_nonlin_fluid.set_data(
                x_nonlin_fluid[idx_nonlin_fluid], y_nonlin_fluid[idx_nonlin_fluid]
            )

            return (
                line_lin_no_fluid,
                line_lin_fluid,
                line_nonlin_no_fluid,
                line_nonlin_fluid,
            )

        anim = FuncAnimation(
            fig=fig,
            func=animate,
            frames=len(common_times),
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


if __name__ == "__main__":
    main()
