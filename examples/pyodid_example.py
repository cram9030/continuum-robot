import numpy as np
from scipy.integrate import solve_ivp
from continuum_robot.models.dynamic_beam_model import DynamicEulerBernoulliBeam
import tempfile
import json

# Simulation parameters
T_FINAL = 0.5  # seconds
DT = 0.01  # Time step for animation
N_SEGMENTS = 6  # Number of beam segments


def create_beam_parameters():
    """Create CSV file with beam parameters."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
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

        params = [
            (length, E, MInertia, rho, A, beam_type, bc)
            for beam_type, bc in [("linear", "FIXED")] + [("linear", "NONE")] * 5
        ]

        for p in params:
            f.write(f"{','.join(str(x) for x in p)}\n")

        return f.name


def run_simulation():
    """Run beam simulation and return results as JSON."""
    # Create parameter file
    param_file = create_beam_parameters()

    # Initialize beam model
    beam = DynamicEulerBernoulliBeam(param_file)
    beam.create_system_func()
    beam.create_input_func()

    # Setup initial conditions
    n_states = beam.linear_model.M.shape[0]
    x0 = np.zeros(2 * n_states)

    # Define time points
    t_span = (0, T_FINAL)
    t_eval = np.arange(t_span[0], t_span[1], DT)

    # Input force function
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

    # Extract beam shapes
    dx = beam.linear_model.get_length() / N_SEGMENTS
    n_pos = len(sol.y) // 2
    n_points = N_SEGMENTS + 1

    # Initialize arrays
    x = np.zeros((len(sol.t), n_points))
    y = np.zeros((len(sol.t), n_points))

    # Calculate beam positions
    for i in range(len(sol.t)):
        pos = sol.y[n_pos::2, i]
        x[i, 0] = 0
        y[i, 0] = 0

        for j in range(N_SEGMENTS):
            x[i, j + 1] = x[i, j] + dx
            y[i, j + 1] = pos[j]

    # Format results for JSON
    results = {
        "times": sol.t.tolist(),
        "x_coords": x.tolist(),
        "y_coords": y.tolist(),
        "tip_displacement": y[:, -1].tolist(),
    }

    return json.dumps(results)
