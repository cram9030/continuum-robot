import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile
import os

from models.dynamic_beam_model import DynamicEulerBernoulliBeam

# Simulation parameters
T_FINAL = 1.0  # seconds
DT = 0.01  # Time step for animation
N_SEGMENTS = 6  # Number of beam segments


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
            for beam_type, bc in [("linear", "FIXED")] + [("linear", "NONE")] * 5
        ]

        for p in params:
            f.write(f"{','.join(str(x) for x in p)}\n")

        linear_file = f.name

    # Create identical file for nonlinear beam
    nonlinear_file = linear_file.replace(".csv", "_nonlinear.csv")
    with open(linear_file, "r") as src, open(nonlinear_file, "w") as dst:
        for line in src:
            dst.write(line.replace("linear", "nonlinear"))

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


def solve_beam_dynamics(beam, x0, t_span, dt):
    """Solve beam dynamics using solve_ivp."""
    # Create time points for solution
    t_eval = np.arange(t_span[0], t_span[1], dt)

    # Zero input force
    def u(t):
        u = np.zeros(len(x0) // 2)
        if t < 0.01:
            u[-2] = 0.1  # Impulse at tip
        return u

    # Solve system
    sol = solve_ivp(
        lambda t, x: beam.get_dynamic_system()(t, x, u),
        t_span,
        x0,
        method="RK45",
        t_eval=t_eval,
    )

    return sol


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


def create_animation(t, x_lin, y_lin):
    """Create animation of both beams."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot settings
    ax.set_xlim(0.0, 1.6)
    ax.set_ylim(np.min(y_lin), np.max(y_lin))
    ax.set_xlabel("Beam Length (m)")
    ax.set_ylabel("Displacement (m)")
    ax.set_title("Linear vs Nonlinear Beam Response to Impulse")

    # Initialize lines
    (line_lin,) = ax.plot([], [], "b-", label="Linear")
    ax.legend()

    def animate(frame):
        # Update lines
        line_lin.set_data(x_lin[frame], y_lin[frame])
        return (line_lin,)

    anim = FuncAnimation(fig, animate, frames=len(t), interval=DT, blit=True)

    return anim


def plot_tip_displacement(t, y_lin):
    """Plot tip displacement over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_lin[:, -1], "b-", label="Linear")
    plt.xlabel("Time (s)")
    plt.ylabel("Tip Displacement (m)")
    plt.title("Beam Tip Response to Impulse")
    plt.legend()
    plt.grid(True)


def main():
    # Create parameter files
    linear_file, nonlinear_file = create_beam_parameters()

    try:
        # Initialize beam models
        linear_beam = DynamicEulerBernoulliBeam(linear_file)
        nonlinear_beam = DynamicEulerBernoulliBeam(nonlinear_file)

        # Create system functions
        for beam in [linear_beam, nonlinear_beam]:
            beam.create_system_func()
            beam.create_input_func()

        # Set up and solve
        t_span = (0, T_FINAL)
        x0_lin = setup_initial_conditions(linear_beam)

        sol_lin = solve_beam_dynamics(linear_beam, x0_lin, t_span, DT)

        # Extract beam shapes
        x_lin, y_lin = extract_beam_shapes(
            sol_lin, N_SEGMENTS, linear_beam.linear_model.get_length() / N_SEGMENTS
        )

        # Create visualization
        create_animation(sol_lin.t, x_lin, y_lin)
        plot_tip_displacement(sol_lin.t, y_lin)

        plt.show()

    finally:
        # Cleanup temporary files
        os.unlink(linear_file)
        os.unlink(nonlinear_file)


if __name__ == "__main__":
    main()
