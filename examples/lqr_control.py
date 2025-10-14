"""
LQR Control Example for Continuum Robot Beams with Kalman Filter Estimation.

This example demonstrates Linear Quadratic Regulator (LQR) control applied to
continuum robot beams with and without state estimation using Kalman filtering.
It compares four scenarios:
1. Uncontrolled system
2. LQR with full state feedback
3. LQR with Kalman filter estimation (noisy measurements)
4. LQR with perfect state feedback for comparison

The Kalman filter estimates the full state from noisy tip position measurements
with measurement noise standard deviation of 0.2 cm.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from multiprocessing import cpu_count, Pool
import time

from continuum_robot.models.dynamic_beam_model import DynamicEulerBernoulliBeam
from continuum_robot.models.force_params import ForceParams
from control_design.linear_quadratic_regulator import LinearQuadraticRegulator
from continuum_robot.control.full_state_linear import FullStateLinear
from estimator_design.linear_quadratic_estimator import LinearQuadraticEstimator
from continuum_robot.estimator.lqe_filter import KalmanFilterLTI

from example_utilities import (
    create_beam_parameters,
    extract_beam_shapes,
    cleanup_temp_files,
    print_performance_table,
    get_material_properties,
    DT,
    N_SEGMENTS,
)

T_FINAL = 0.25
# Time step for discretization
dt = 0.001  # 1ms for discrete updates

# Visualization styling - shared across all plotting functions
SCENARIO_COLORS = {
    "Uncontrolled": "red",
    "LQR (Full State)": "blue",
    "LQR + Discrete LQE": "green",
    "LQR (Perfect Sensor)": "purple",
}
SCENARIO_STYLES = {
    "Uncontrolled": "-",
    "LQR (Full State)": "-",
    "LQR + Discrete LQE": "-",
    "LQR (Perfect Sensor)": ":",
}


def create_impulse_function(n_dofs, amplitude=1.0, start_time=0.0, duration=0.01):
    """Create impulse input function for disturbance testing."""

    def impulse(t):
        """Apply impulse force at beam tip."""
        u_vec = np.zeros(n_dofs)
        if start_time <= t < start_time + duration:
            # Apply transverse force at tip (last w DOF)
            u_vec[-2] = amplitude
        return u_vec

    return impulse


def design_lqr_controller(beam, discrete=False, dt=0.001):
    """Design LQR controller for the beam system.

    Args:
        beam: Beam model
        discrete: Whether to design discrete-time or continuous-time controller
        dt: Time step for discrete-time controller (default: 1ms = 0.001s)

    Returns:
        Tuple of (FullStateLinear controller, LinearQuadraticRegulator object)
    """
    controller_type = "Discrete-Time" if discrete else "Continuous-Time"
    print(f"Designing {controller_type} LQR Controller...")

    # Extract linear system matrices from the beam
    K_beam = beam.beam_model.get_stiffness_matrix()
    M_beam = beam.beam_model.get_mass_matrix()

    print(f"Beam system: {K_beam.shape[0]} DOFs")

    # Design LQR controller
    n_dofs = K_beam.shape[0]

    # Construct A matrix for state-space representation
    # For structural dynamics: M*q̈ + K*q = u
    # State space form with x = [q, q̇]:
    # dx/dt = [0      I   ] x + [0    ] u
    #         [-M^-1*K  0 ]     [M^-1*B]
    A_cont = np.zeros((2 * n_dofs, 2 * n_dofs))
    A_cont[:n_dofs, n_dofs:] = np.eye(n_dofs)  # Upper right: I
    M_inv = np.linalg.inv(M_beam)
    A_cont[n_dofs:, :n_dofs] = -M_inv @ K_beam  # Lower left: -M^-1*K

    # Construct B matrix with actuation only at specific DOFs
    # Actuation applied at indices -3 (axial) and -2 (transverse) at tip
    B_input = np.zeros((n_dofs, 2))
    B_input[-3, 0] = 1.0  # Axial displacement at tip
    B_input[-2, 1] = 1.0  # Transverse displacement at tip

    B_cont = np.zeros((2 * n_dofs, 2))
    B_cont[n_dofs:, :] = M_inv @ B_input  # Lower half: M^-1 * B_input

    # Discretize if needed
    if discrete:
        import control as ct

        sys_cont = ct.StateSpace(
            A_cont, B_cont, np.eye(2 * n_dofs), np.zeros((2 * n_dofs, 2))
        )
        sys_disc = ct.sample_system(sys_cont, dt, method="zoh")
        A = np.array(sys_disc.A)
        B = np.array(sys_disc.B)
    else:
        A = A_cont
        B = B_cont

    print(f"  - A matrix shape: {A.shape}")
    print(f"  - B matrix shape: {B.shape}")
    print(f"  - Actuation DOFs: {[-3, -2]} (tip axial and transverse displacement)")
    if discrete:
        print(f"  - Sample time: {dt*1000:.2f}ms")

    # Weighting matrices
    # Q: State weighting (emphasize position control)
    Q = np.eye(2 * n_dofs)
    Q[2:n_dofs:3, 2:n_dofs:3] *= 100  # Position weighting
    Q[n_dofs:, n_dofs:] *= 10  # Velocity weighting

    # R: Control weighting (penalize control effort)
    R = np.eye(2) * 1.0  # 2 inputs now

    # Create LQR controller
    lqr = LinearQuadraticRegulator(A, B, Q, R)
    if discrete:
        lqr.set_discrete_time(True)
    gain_matrix, _ = lqr.compute_gain_matrix()

    # Verify stability
    A_cl = A - B @ gain_matrix
    eigenvals = np.linalg.eigvals(A_cl)

    if discrete:
        max_magnitude = np.max(np.abs(eigenvals))
        is_stable = max_magnitude < 1.0
        print(f"  - Max closed-loop eigenvalue magnitude: {max_magnitude:.6f}")
    else:
        max_real_part = np.max(np.real(eigenvals))
        is_stable = max_real_part < 0
        print(f"  - Max closed-loop eigenvalue real part: {max_real_part:.6f}")

    print(f"{controller_type} LQR Controller designed:")
    print(f"  - Gain matrix shape: {gain_matrix.shape}")
    print(f"  - System is {'stable' if is_stable else 'unstable'}")

    return (
        FullStateLinear(
            gain_matrix, B, is_discrete=discrete, dt=dt if discrete else None
        ),
        lqr,
    )


def design_lqe_filter(
    beam, lqr_controller, measurement_noise_std=0.002, discrete=False, dt=0.001
):
    """
    Design LQE Kalman filter for beam tip position estimation.

    This filter uses Linear Quadratic Estimator (LQE) design to compute optimal
    steady-state estimator gains, then uses an LTI filter for efficient state estimation.

    Args:
        beam: Beam model
        lqr_controller: LQR controller object containing A and B matrices
        measurement_noise_std: Measurement noise standard deviation (default: 0.2cm = 0.002m)
        discrete: Whether to design discrete-time or continuous-time filter
        dt: Time step for discrete-time filter (default: 1ms = 0.001s)

    Returns:
        KalmanFilterLTI: Configured LQE-based Kalman filter
    """
    filter_type = "Discrete-Time" if discrete else "Continuous-Time"
    print(f"Designing {filter_type} LQE Kalman Filter...")

    # Get system matrices from LQR controller
    A = lqr_controller.get_A()
    B_cont = lqr_controller.get_B()
    K = lqr_controller.get_K()

    n_states = A.shape[0]
    n_dofs = n_states // 2

    B = np.zeros((n_states, n_dofs))
    B[n_dofs:, :] = np.linalg.inv(beam.beam_model.get_mass_matrix())  # Lower half: M^-1

    # Closed-loop dynamics: A_cl = A - B*K
    # Use closed-loop A matrix for filter (system under control)
    A_cl = A - B_cont @ K

    # Observation matrix C - measure multiple DOFs
    n_measurements = 4

    C = np.zeros((n_measurements, n_states))
    C[0, 17] = 1
    C[1, :] = A[17, :]
    C[2, :] = A[-3, :]
    C[3, :] = A[-2, :]

    # Measurement noise covariance
    R = measurement_noise_std**2 * np.eye(n_measurements)

    # Process noise covariance Q
    Q = 1e-3 * np.eye(n_states)

    # Design LQE to compute optimal steady-state estimator gain
    print("  - Computing optimal estimator gain using LQE...")
    lqe = LinearQuadraticEstimator(A=A_cl, C=C, Q=Q, R=R)
    if discrete:
        lqe.set_discrete_time(True)
    L, P = lqe.compute_estimator_gain()

    # Initial state estimate (zero)
    x0 = np.zeros(n_states)

    # Get tip state description
    tip_dof_idx = n_dofs - 2
    tip_param, tip_node = beam.beam_model.get_dof_to_node_param(tip_dof_idx)
    tip_state = f"{tip_param}{tip_node}"

    print(f"{filter_type} LQE Filter configured:")
    print(f"  - Type: {filter_type} LTI Kalman Filter")
    print(f"  - Primary measurement: tip position ({tip_state})")
    print(f"  - Measurement noise std: {measurement_noise_std*1000:.1f}mm")
    print(f"  - State dimension: {n_states}")
    print("  - Estimator gain L computed via LQE")
    print("  - Using steady-state gains (no covariance updates)")
    if discrete:
        print(f"  - Sample time: {dt*1000:.2f}ms")

    return KalmanFilterLTI(A_cl, B, C, L, P, x0, dt=dt if discrete else None)


def simulate_control_scenario(task):
    """
    Simulate a single control scenario (uncontrolled, LQR, LQR+Kalman).

    Args:
        task: Dictionary containing simulation configuration

    Returns:
        Tuple: (case_name, solution, computation_time, solver_stats)
    """
    case_name = task["case_name"]
    param_file = task["param_file"]
    controller = task.get("controller", None)
    kalman_filter = task.get("kalman_filter", None)
    impulse_amplitude = task.get("impulse_amplitude", 10.0)
    measurement_noise_std = task.get("measurement_noise_std", 0.002)

    print(f"Starting simulation: {case_name}")
    start_time = time.time()

    # Create beam model
    force_params = ForceParams(enable_gravity_effects=True, enable_fluid_effects=False)
    beam = DynamicEulerBernoulliBeam(param_file, force_params)
    beam.create_system_func()
    beam.create_input_func()

    # Get system dimensions
    n_states = beam.beam_model.M.shape[0]
    n_full_states = 2 * n_states
    x0 = np.zeros(n_full_states)

    # Create impulse function
    impulse_func = create_impulse_function(
        n_states, amplitude=impulse_amplitude, start_time=0.1, duration=0.0001
    )

    # Initialize random seed for repeatable noisy measurements
    np.random.seed(hash(case_name) % 2**32)

    # Control and estimation log: stores estimated states and control inputs at discrete intervals
    # Structure: {time: {'estimated_state': array, 'control_input': array}}
    control_estimate_log = {}

    # Fetch gravity force from force registry for Kalman filter
    gravity_force = None
    registered_forces = beam.force_registry.get_registered_forces()
    for force in registered_forces:
        if force.get_name() == "GravityForce":
            gravity_force = force
            break

    def system_with_control_estimation(t, x):
        """System dynamics with control and estimation."""

        # External disturbance
        disturbance = impulse_func(t)

        if controller is None:
            # Uncontrolled case
            control_force = np.zeros(n_states)
        elif kalman_filter is None:
            # Perfect state feedback case
            reference = np.zeros_like(x)
            control_force = controller.compute_input(x, reference, t)
        else:
            # Kalman filter estimation case (continuous-time LQE filter)
            # Get control input for estimation (includes gravity and previous control)
            if gravity_force is not None:
                gravity_input = gravity_force.compute_forces(x, t)
                control_for_estimation = gravity_input
            else:
                control_for_estimation = np.zeros(n_states)

            # Continuous-time LQE filter: call estimate_states at every time step
            n_measurements = 4

            # Create noisy measurements for all monitored DOFs
            noise = np.random.normal(0, measurement_noise_std, size=(n_measurements,))
            y = kalman_filter.C @ x + noise

            # Continuous-time filter integrates between measurements
            estimated_state = kalman_filter.estimate_states(
                y, control_for_estimation, t
            )

            # Compute control input using estimated state
            reference = np.zeros_like(estimated_state)
            control_force = controller.compute_input(x, reference, t)

            # Log estimate and control for visualization
            control_estimate_log[t] = {
                "estimated_state": estimated_state.copy(),
                "control_input": control_force.copy(),
            }

        # Combined input: disturbance + control
        total_input = disturbance + control_force

        # System dynamics
        return beam.get_dynamic_system()(t, x, total_input)

    # Solve system
    t_span = (0, T_FINAL)
    t_eval = np.arange(0, T_FINAL, DT)

    solution = solve_ivp(
        system_with_control_estimation,
        t_span,
        x0,
        method="LSODA",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
    )

    computation_time = time.time() - start_time

    # Extract solver statistics
    solver_stats = {
        "nfev": solution.nfev if hasattr(solution, "nfev") else 0,
        "njev": solution.njev if hasattr(solution, "njev") else 0,
        "nlu": solution.nlu if hasattr(solution, "nlu") else 0,
    }

    print(f"  - {case_name} completed: {solution.message}")
    print(f"  - Computation time: {computation_time:.3f}s")
    print(f"  - Function evaluations: {solution.nfev}")

    return case_name, solution, computation_time, solver_stats, control_estimate_log


def plot_displacement_components(solutions):
    """Create three separate plots for axial (u), transverse (w), and rotation (phi) displacements over time.

    Each plot shows all nodes' displacement components vs time for all scenarios.

    Args:
        solutions: Dictionary mapping case_name -> solution object
    """
    print("Creating displacement component plots...")

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    for case_name, solution in solutions.items():
        y_array = np.array(solution.y) if isinstance(solution.y, list) else solution.y
        n_pos = len(y_array) // 2

        # Extract all displacement components over time
        # DOF ordering: [u1, w1, φ1, u2, w2, φ2, ...]
        u_indices = list(range(0, n_pos, 3))  # indices 0, 3, 6, 9, ...
        w_indices = list(range(1, n_pos, 3))  # indices 1, 4, 7, 10, ...
        phi_indices = list(range(2, n_pos, 3))  # indices 2, 5, 8, 11, ...

        color = SCENARIO_COLORS.get(case_name, "black")
        style = SCENARIO_STYLES.get(case_name, "-")

        # Plot axial displacements for all nodes over time
        for i, idx in enumerate(u_indices):
            label = f"{case_name} (Node {i})" if i == len(u_indices) - 1 else None
            alpha = 0.3 + 0.7 * (
                i / max(len(u_indices) - 1, 1)
            )  # Gradient alpha for visibility
            ax1.plot(
                solution.t,
                y_array[idx, :],
                color=color,
                linestyle=style,
                linewidth=2,
                alpha=alpha,
                label=label,
            )

        # Plot transverse displacements for all nodes over time
        for i, idx in enumerate(w_indices):
            label = f"{case_name} (Node {i})" if i == len(w_indices) - 1 else None
            alpha = 0.3 + 0.7 * (i / max(len(w_indices) - 1, 1))
            ax2.plot(
                solution.t,
                y_array[idx, :],
                color=color,
                linestyle=style,
                linewidth=2,
                alpha=alpha,
                label=label,
            )

        # Plot rotations for all nodes over time
        for i, idx in enumerate(phi_indices):
            label = f"{case_name} (Node {i})" if i == len(phi_indices) - 1 else None
            alpha = 0.3 + 0.7 * (i / max(len(phi_indices) - 1, 1))
            ax3.plot(
                solution.t,
                y_array[idx, :],
                color=color,
                linestyle=style,
                linewidth=2,
                alpha=alpha,
                label=label,
            )

    # Configure axial displacement plot
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Axial Displacement u (m)")
    ax1.set_title("Axial Displacement vs Time (All Nodes)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Configure transverse displacement plot
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Transverse Displacement w (m)")
    ax2.set_title("Transverse Displacement vs Time (All Nodes)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # Configure rotation plot
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Rotation φ (rad)")
    ax3.set_title("Rotation vs Time (All Nodes)")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_multi_scenario_animation(solutions, estimate_logs):  # noqa: C901
    """Create animation comparing multiple control scenarios.

    Args:
        solutions: Dictionary mapping case_name -> solution object
        estimate_logs: Dictionary mapping case_name -> control_estimate_log dictionary
                      where each log entry contains {'estimated_state': array, 'control_input': array}
    """
    print("Creating multi-scenario comparison animation...")

    # Extract beam shapes for all solutions (actual states)
    props = get_material_properties()
    dx = props["length"]

    scenario_data = {}
    estimate_data = {}
    all_y_values = []

    for case_name, solution in solutions.items():
        x_coords, y_coords = extract_beam_shapes(solution, N_SEGMENTS, dx, linear=True)
        scenario_data[case_name] = (x_coords, y_coords)
        all_y_values.extend(y_coords.flatten())

        # Extract estimated beam shapes if available
        if case_name in estimate_logs and estimate_logs[case_name]:
            # Create pseudo-solution object from control_estimate_log for beam shape extraction
            control_estimate_log = estimate_logs[case_name]
            times = sorted(control_estimate_log.keys())

            # Build state array from logged estimates
            first_entry = (
                list(control_estimate_log.values())[0] if control_estimate_log else None
            )
            n_states = len(first_entry["estimated_state"]) if first_entry else 0
            if n_states > 0:
                est_states = np.array(
                    [control_estimate_log[t]["estimated_state"] for t in times]
                ).T

                # Create a pseudo-solution structure
                class EstimateSolution:
                    def __init__(self, t, y):
                        self.t = np.array(t)
                        self.y = y

                # Interpolate estimates to match solution time points
                est_interp = np.zeros((n_states, len(solution.t)))
                for i in range(n_states):
                    est_interp[i, :] = np.interp(solution.t, times, est_states[i, :])

                est_sol = EstimateSolution(solution.t, est_interp)
                x_est, y_est = extract_beam_shapes(est_sol, N_SEGMENTS, dx, linear=True)
                estimate_data[case_name] = (x_est, y_est)
                all_y_values.extend(y_est.flatten())

    # Create animation with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20))

    # Setup animation plot
    ax1.set_xlim(0.0, 1.6)

    # Set y-axis limits with safety checks
    if len(all_y_values) > 0:
        y_min, y_max = np.min(all_y_values), np.max(all_y_values)
        y_range = y_max - y_min
        if y_range < 1e-10:  # All values essentially the same
            y_center = (y_max + y_min) / 2
            ax1.set_ylim(y_center - 0.01, y_center + 0.01)
        else:
            y_pad = y_range * 0.1
            ax1.set_ylim(y_min - y_pad, y_max + y_pad)
    else:
        ax1.set_ylim(-0.02, 0.02)

    ax1.set_xlabel("Beam Length (m)")
    ax1.set_ylabel("Displacement (m)")
    ax1.set_title("Control and Estimation Scenarios Comparison")
    ax1.grid(True)

    # Create line objects for animation
    lines = {}
    estimate_lines = {}

    for case_name in scenario_data.keys():
        color = SCENARIO_COLORS.get(case_name, "black")
        style = SCENARIO_STYLES.get(case_name, "-")

        # Actual state line
        (line,) = ax1.plot(
            [],
            [],
            color=color,
            linestyle=style,
            linewidth=2,
            label=f"{case_name} (Actual)",
        )
        lines[case_name] = line

        # Estimated state line (if available)
        if case_name in estimate_data:
            (est_line,) = ax1.plot(
                [],
                [],
                color=color,
                linestyle=":",
                linewidth=2,
                alpha=0.7,
                label=f"{case_name} (Estimate)",
            )
            estimate_lines[case_name] = est_line

    ax1.legend(loc="best")

    # Setup tip displacement comparison
    tip_responses = {}
    tip_estimates = {}

    for case_name, solution in solutions.items():
        y_array = np.array(solution.y) if isinstance(solution.y, list) else solution.y
        n_pos = len(y_array) // 2
        tip_displacement = y_array[n_pos - 2, :]  # Tip transverse displacement
        tip_responses[case_name] = tip_displacement

        color = SCENARIO_COLORS.get(case_name, "black")
        style = SCENARIO_STYLES.get(case_name, "-")

        # Plot actual tip displacement
        ax2.plot(
            solution.t,
            tip_displacement,
            color=color,
            linestyle=style,
            label=f"{case_name} (Actual)",
            linewidth=2,
        )

        # Plot estimated tip displacement if available
        if case_name in estimate_data:
            x_est, y_est = estimate_data[case_name]
            tip_est = y_est[:, -1]  # Last node is the tip
            tip_estimates[case_name] = tip_est
            ax2.plot(
                solution.t,
                tip_est,
                color=color,
                linestyle=":",
                label=f"{case_name} (Estimate)",
                linewidth=2,
                alpha=0.7,
            )

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Tip Displacement (m)")
    ax2.set_title("Tip Response Comparison")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # Setup control input comparison
    for case_name in estimate_logs.keys():
        control_estimate_log = estimate_logs[case_name]

        if control_estimate_log:
            # Extract times and control inputs from the log
            times = sorted(control_estimate_log.keys())

            # Get control input dimension from first entry
            first_entry = (
                list(control_estimate_log.values())[0] if control_estimate_log else None
            )
            if first_entry and "control_input" in first_entry:
                n_control = len(first_entry["control_input"])

                # Extract control inputs for each DOF
                # We'll plot the two most significant control inputs (tip axial and transverse)
                if n_control >= 2:
                    # Axial control (DOF -3, index -3)
                    axial_control = [
                        control_estimate_log[t]["control_input"][-3] for t in times
                    ]
                    # Transverse control (DOF -2, index -2)
                    transverse_control = [
                        control_estimate_log[t]["control_input"][-2] for t in times
                    ]

                    color = SCENARIO_COLORS.get(case_name, "black")

                    # Plot transverse control input (typically more significant)
                    ax3.plot(
                        times,
                        transverse_control,
                        color=color,
                        linestyle="-",
                        label=f"{case_name} (Transverse)",
                        linewidth=2,
                    )

                    # Optionally also plot axial control with dashed line
                    ax3.plot(
                        times,
                        axial_control,
                        color=color,
                        linestyle="--",
                        label=f"{case_name} (Axial)",
                        linewidth=1.5,
                        alpha=0.7,
                    )

    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Control Input (N)")
    ax3.set_title("Control Input Comparison")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    # Performance metrics removed from plot - will be printed to terminal instead

    def animate(frame):
        """Animation function for multiple scenarios."""
        animated_objects = []

        # Animate actual states
        for case_name, (x_coords, y_coords) in scenario_data.items():
            if frame < len(x_coords) and frame < len(y_coords):
                lines[case_name].set_data(x_coords[frame], y_coords[frame])
                animated_objects.append(lines[case_name])

        # Animate estimated states
        for case_name, (x_est, y_est) in estimate_data.items():
            if frame < len(x_est) and frame < len(y_est):
                estimate_lines[case_name].set_data(x_est[frame], y_est[frame])
                animated_objects.append(estimate_lines[case_name])

        return animated_objects

    # Create animation
    min_frames = min(len(coords[0]) for coords in scenario_data.values())
    anim = FuncAnimation(
        fig, animate, frames=min_frames, interval=DT * 1000, blit=True, repeat=True
    )

    plt.tight_layout()
    return anim


def main(run_scenarios=None):
    """Main function to demonstrate LQR control with and without LQE estimation.

    Args:
        run_scenarios: List of scenario names to run. If None, runs all scenarios.
                      Valid names: 'uncontrolled', 'lqr', 'lqe_discrete'
    """
    if run_scenarios is None:
        run_scenarios = ["uncontrolled", "lqr", "lqe_discrete"]

    print("=" * 80)
    print("DISCRETE-TIME LQR CONTROL WITH DISCRETE-TIME LQE FILTER COMPARISON")
    print("=" * 80)
    print("Available scenarios:")
    print("1. Uncontrolled system")
    print("2. Discrete-time LQR with full state feedback (ideal case)")
    print("3. Discrete-time LQR with discrete-time LQE filter estimation")
    print(f"\nRunning scenarios: {', '.join(run_scenarios)}")
    print(f"Running on {cpu_count()} CPU cores")
    print(f"System: {N_SEGMENTS} segments, {T_FINAL}s simulation")
    print("-" * 80)

    # Create beam parameter files
    linear_file, _, _ = create_beam_parameters()

    try:
        # Create reference beam for controller design
        force_params = ForceParams(
            enable_gravity_effects=True, enable_fluid_effects=False
        )
        reference_beam = DynamicEulerBernoulliBeam(linear_file, force_params)
        reference_beam.create_system_func()
        reference_beam.create_input_func()

        # Design discrete-time LQR controller and discrete-time LQE filter
        print("\nDesigning control and estimation systems...")
        controller, lqr_controller = design_lqr_controller(
            reference_beam, discrete=True, dt=dt
        )

        # Design discrete-time LQE filter with 1ms sample time
        lqe_filter_discrete = design_lqe_filter(
            reference_beam, lqr_controller, discrete=True, dt=dt
        )

        # Define simulation scenarios
        measurement_noise_std = 0.002  # 0.2cm
        impulse_amplitude = 0.01

        # Build simulation tasks based on requested scenarios
        simulation_tasks = []

        if "uncontrolled" in run_scenarios:
            simulation_tasks.append(
                {
                    "case_name": "Uncontrolled",
                    "param_file": linear_file,
                    "controller": None,
                    "kalman_filter": None,
                    "impulse_amplitude": impulse_amplitude,
                    "measurement_noise_std": measurement_noise_std,
                }
            )

        if "lqr" in run_scenarios:
            simulation_tasks.append(
                {
                    "case_name": "LQR (Full State)",
                    "param_file": linear_file,
                    "controller": controller,
                    "kalman_filter": None,
                    "impulse_amplitude": impulse_amplitude,
                    "measurement_noise_std": measurement_noise_std,
                }
            )

        if "lqe_discrete" in run_scenarios:
            simulation_tasks.append(
                {
                    "case_name": "LQR + Discrete LQE",
                    "param_file": linear_file,
                    "controller": controller,
                    "kalman_filter": lqe_filter_discrete,
                    "impulse_amplitude": impulse_amplitude,
                    "measurement_noise_std": measurement_noise_std,
                }
            )

        print(f"\nStarting parallel simulation of {len(simulation_tasks)} scenarios...")
        start_time = time.time()

        # Run simulations in parallel using multiprocessing
        with Pool(processes=min(len(simulation_tasks), cpu_count())) as pool:
            results = pool.map(simulate_control_scenario, simulation_tasks)

        total_time = time.time() - start_time
        print(f"\nAll simulations completed in {total_time:.2f} seconds")

        # Process results
        solutions = {}
        computation_times = {}
        solver_statistics = {}
        estimate_logs = {}

        for (
            case_name,
            solution,
            comp_time,
            solver_stats,
            control_estimate_log,
        ) in results:
            solutions[case_name] = solution
            computation_times[case_name] = comp_time
            solver_statistics[case_name] = solver_stats
            estimate_logs[case_name] = control_estimate_log

        # Print performance comparison table
        print_performance_table(computation_times, solver_statistics)

        # Print control performance comparison
        print("\n" + "=" * 80)
        print("CONTROL AND ESTIMATION PERFORMANCE ANALYSIS")
        print("=" * 80)

        uncontrolled_solution = solutions.get("Uncontrolled")
        lqr_solution = solutions.get("LQR (Full State)")
        lqe_solution = solutions.get("LQR + Discrete LQE")

        if all(
            sol is not None
            for sol in [uncontrolled_solution, lqr_solution, lqe_solution]
        ):
            # Extract tip responses for analysis
            def get_tip_response(solution):
                y_array = (
                    np.array(solution.y) if isinstance(solution.y, list) else solution.y
                )
                n_pos = len(y_array) // 2
                return y_array[n_pos - 2, :]  # Tip transverse displacement

            tip_uncontrolled = get_tip_response(uncontrolled_solution)
            tip_lqr = get_tip_response(lqr_solution)
            tip_lqe = get_tip_response(lqe_solution)

            # Calculate performance metrics
            peak_uncontrolled = np.max(np.abs(tip_uncontrolled))
            peak_lqr = np.max(np.abs(tip_lqr))
            peak_lqe = np.max(np.abs(tip_lqe))

            print("Peak Displacement:")
            print(f"  Uncontrolled:        {peak_uncontrolled:.3f}m")
            print(
                f"  LQR (Full State):    {peak_lqr:.3f}m ({(peak_uncontrolled-peak_lqr)/peak_uncontrolled*100:+.1f}%)"
            )
            print(
                f"  LQR + Discrete LQE:  {peak_lqe:.3f}m ({(peak_uncontrolled-peak_lqe)/peak_uncontrolled*100:+.1f}%)"
            )

            print("\nEstimation vs Perfect Feedback:")
            print(
                f"  Discrete LQE filter degradation: {(peak_lqe-peak_lqr)/peak_lqr*100:+.1f}%"
            )

        # Create displacement component plots
        print("\nCreating displacement component plots...")
        _ = plot_displacement_components(solutions)  # noqa: F841

        # Create and display animation
        print("\nCreating comparison animation...")
        _ = create_multi_scenario_animation(solutions, estimate_logs)  # noqa: F841
        plt.show()

        print("\nSimulation completed successfully!")
        print("Close the animation window to exit.")

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        cleanup_temp_files(linear_file)


if __name__ == "__main__":
    import sys

    # Parse command line arguments to select which scenarios to run
    # Usage: python lqr_control.py [uncontrolled] [lqr] [lqe_discrete]
    # Example: python lqr_control.py lqe_discrete           (runs only discrete LQE filter)
    # Example: python lqr_control.py lqr lqe_discrete      (runs LQR and discrete LQE filter)
    # No arguments: runs all scenarios

    valid_scenarios = {"uncontrolled", "lqr", "lqe_discrete"}

    if len(sys.argv) > 1:
        # Parse scenario names from command line
        requested_scenarios = [arg.lower() for arg in sys.argv[1:]]

        # Validate scenarios
        invalid_scenarios = [s for s in requested_scenarios if s not in valid_scenarios]
        if invalid_scenarios:
            print(f"Error: Invalid scenario(s): {', '.join(invalid_scenarios)}")
            print(f"Valid scenarios: {', '.join(sorted(valid_scenarios))}")
            print("\nUsage examples:")
            print("  python lqr_control.py                         # Run all scenarios")
            print(
                "  python lqr_control.py lqe_discrete            # Run discrete LQE filter only"
            )
            print(
                "  python lqr_control.py lqr lqe_discrete        # Run LQR and discrete LQE filter"
            )
            print(
                "  python lqr_control.py uncontrolled lqr        # Run uncontrolled and LQR"
            )
            sys.exit(1)

        run_scenarios = requested_scenarios
    else:
        # No arguments: run all scenarios
        run_scenarios = None

    main(run_scenarios=run_scenarios)
