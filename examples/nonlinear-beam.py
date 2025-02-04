import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile
import os

from models.dynamic_beam_model import DynamicEulerBernoulliBeam

# Simulation parameters
T_FINAL = 0.1  # seconds
DT = 0.01  # Time step for animation
N_SEGMENTS = 4  # Number of beam segments


def create_beam_parameters():
    """Create CSV file with beam parameters for both models."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        # Write header
        f.write(
            "length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition\n"
        )

        # Nitinol parameters
        length = 0.25  # Each segment length for 1.5m total
        E = 75e9  # Young's modulus (Pa)
        r = 0.005  # Radius (m)
        MInertia = np.pi * r**4 / 4  # Moment of inertia
        rho = 6450  # Density (kg/m³)
        A = np.pi * r**2  # Cross-sectional area

        # Create 6 segments, first has fixed boundary condition
        params = [
            (length, E, MInertia, rho, A, beam_type, bc)
            for beam_type, bc in [("nonlinear", "FIXED")] + [("nonlinear", "NONE")] * 5
        ]

        for p in params:
            f.write(f"{','.join(str(x) for x in p)}\n")

        nonlinear_file = f.name

    return nonlinear_file


def setup_initial_conditions(beam):
    """Set up initial conditions."""
    n_states = beam.nonlinear_model.M.shape[0]

    # Zero initial positions and velocities except tip velocity
    x0 = np.zeros(2 * n_states)

    return x0


def solve_beam_dynamics(beam, x0, t_span, dt):
    """Solve beam dynamics using solve_ivp."""
    # Create time points for solution

    def u_nonlinear(t):
        return np.sin(t) * np.ones(len(x0) // 2)

    sol_nonlinear = solve_ivp(lambda t, x: beam(t, x, u_nonlinear), t_span, x0)

    return sol_nonlinear


def extract_beam_shapes(sol, n_segments, dx):
    """Extract beam x,y coordinates over time."""
    n_pos = len(sol.y) // 2
    n_points = n_segments + 1

    # Initialize arrays
    x = np.zeros((len(sol.t), n_points))
    y = np.zeros((len(sol.t), n_points))

    # For each time step
    for i in range(len(sol.t)):
        # Get positions at this time
        pos = sol.y[n_pos::2, i]

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
    nonlinear_file = create_beam_parameters()

    try:
        # Initialize beam models
        nonlinear_beam = DynamicEulerBernoulliBeam(nonlinear_file)

        # Create system functions
        nonlinear_beam.create_system_func()
        nonlinear_beam.create_input_func()

        # Set up and solve
        t_span = (0, T_FINAL)
        x0_lin = setup_initial_conditions(nonlinear_beam)

        sol_lin = solve_beam_dynamics(
            nonlinear_beam.get_dynamic_system(), x0_lin, t_span, DT
        )

        # Extract beam shapes
        x_lin, y_lin = extract_beam_shapes(
            sol_lin,
            N_SEGMENTS,
            nonlinear_beam.nonlinear_model.get_length() / N_SEGMENTS,
        )

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Animation plot
        ax1.set_xlim(0.0, 1.1)
        # Add padding to y limits
        y_pad = (np.max(y_lin) - np.min(y_lin)) * 0.1
        ax1.set_ylim(np.min(y_lin) - y_pad, np.max(y_lin) + y_pad)
        ax1.set_xlabel("Beam Length (m)")
        ax1.set_ylabel("Displacement (m)")
        ax1.set_title("Nonlinear Beam Response")
        ax1.grid(True)

        (line_lin,) = ax1.plot(x_lin[0], y_lin[0], "b-", label="Nonlinear")
        ax1.legend()

        # Tip displacement plot
        ax2.plot(sol_lin.t, y_lin[:, -1], "b-", label="Nonlinear")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Tip Displacement (m)")
        ax2.set_title("Beam Tip Response")
        ax2.grid(True)
        ax2.legend()

        def animate(frame):
            line_lin.set_data(x_lin[frame], y_lin[frame])
            return (line_lin,)

        anim = FuncAnimation(
            fig=fig,
            func=animate,
            frames=len(sol_lin.t),
            interval=DT * 1000,  # Convert to milliseconds
            blit=True,
        )

        plt.tight_layout()
        plt.show()

        return anim  # Return to prevent garbage collection

    finally:
        # Cleanup temporary files
        os.unlink(nonlinear_file)


if __name__ == "__main__":
    main()
