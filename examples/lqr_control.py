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
from multiprocessing import cpu_count
import time

from continuum_robot.models.dynamic_beam_model import DynamicEulerBernoulliBeam
from continuum_robot.models.force_params import ForceParams
from control_design.linear_quadratic_regulator import LinearQuadraticRegulator
from continuum_robot.control.full_state_linear import FullStateLinear
from continuum_robot.estimator.kalman_filter import HybridContinuousDiscreteKalman

from example_utilities import (
    create_beam_parameters,
    extract_beam_shapes,
    cleanup_temp_files,
    print_performance_table,
    get_material_properties,
    T_FINAL,
    DT,
    N_SEGMENTS,
)


def create_impulse_function(n_dofs, amplitude=1.0, duration=0.01):
    """Create impulse input function for disturbance testing."""

    def impulse(t):
        """Apply impulse force at beam tip."""
        u_vec = np.zeros(n_dofs)
        if t < duration:
            # Apply transverse force at tip (last w DOF)
            u_vec[-2] = amplitude
        return u_vec

    return impulse


def design_lqr_controller(beam):
    """Design LQR controller for the beam system."""
    print("Designing LQR Controller...")

    # Extract linear system matrices from the beam
    K_beam = beam.beam_model.get_stiffness_matrix()
    M_beam = beam.beam_model.get_mass_matrix()

    print(f"Beam system: {K_beam.shape[0]} DOFs")

    # Design LQR controller
    n_dofs = K_beam.shape[0]

    # Weighting matrices
    # Q: State weighting (emphasize position control)
    Q = np.eye(2 * n_dofs)
    Q[:n_dofs, :n_dofs] *= 100  # Position weighting
    Q[n_dofs:, n_dofs:] *= 10  # Velocity weighting

    # R: Control weighting (penalize control effort)
    R = np.eye(n_dofs) * 1.0

    # Create LQR controller
    lqr = LinearQuadraticRegulator(K_beam, M_beam, Q, R)
    gain_matrix = lqr.compute_gain_matrix()

    # Verify stability
    A = lqr.get_A()
    B = lqr.get_B()
    A_cl = A - B @ gain_matrix
    eigenvals = np.linalg.eigvals(A_cl)
    max_real_part = np.max(np.real(eigenvals))

    print("LQR Controller designed:")
    print(f"  - Gain matrix shape: {gain_matrix.shape}")
    print(f"  - Max closed-loop eigenvalue real part: {max_real_part:.6f}")
    print(f"  - System is {'stable' if max_real_part < 0 else 'unstable'}")

    return FullStateLinear(gain_matrix), lqr


def design_kalman_filter(beam, lqr_controller, measurement_noise_std=0.002):
    """
    Design Kalman filter for beam tip position estimation.

    Args:
        beam: Beam model
        lqr_controller: LQR controller object containing A and B matrices
        measurement_noise_std: Measurement noise standard deviation (default: 0.2cm = 0.002m)

    Returns:
        HybridContinuousDiscreteKalman: Configured Kalman filter for tip position estimation
    """
    print("Designing Kalman Filter for tip position estimation...")

    # Get system matrices from LQR controller
    A = lqr_controller.get_A()
    B = lqr_controller.get_B()
    K = lqr_controller.get_K()

    n_states = A.shape[0]
    n_dofs = n_states // 2

    # Observation matrix H: measure multiple positions for better observability
    # In the state vector [pos1, pos2, ..., posN, vel1, vel2, ..., velN]
    # Measure positions at several points along the beam (every 2nd DOF + tip)
    n_measurements = min(5, n_dofs)  # Measure up to 5 positions
    measurement_indices = np.linspace(0, n_dofs - 1, n_measurements, dtype=int)

    H = np.zeros((n_measurements, n_states))
    for i, dof_idx in enumerate(measurement_indices):
        H[i, dof_idx] = 1.0  # Measure positions at selected DOFs

    # Get state descriptions for measurement DOFs
    measurement_states = []
    for dof_idx in measurement_indices:
        param, node = beam.beam_model.get_dof_to_node_param(dof_idx)
        measurement_states.append(f"{param}{node}")

    print(f"  - Measuring {n_measurements} positions: {', '.join(measurement_states)}")

    # Update measurement noise covariance for multiple measurements
    R = measurement_noise_std**2 * np.eye(n_measurements)

    # Process noise covariance Q
    # Assume small but non-zero process noise for numerical stability
    Q = 1e-3 * np.eye(n_states)

    # Initial error covariance P
    # Start with moderate uncertainty
    P = 1e-1 * np.eye(n_states)

    # Initial state estimate (zero)
    x0 = np.zeros(n_states)

    # Time step
    dt = 0.0001  # 0.1ms for discrete updates

    # Get tip state description (last transverse displacement)
    tip_dof_idx = n_dofs - 2  # Tip transverse displacement w at last node
    tip_param, tip_node = beam.beam_model.get_dof_to_node_param(tip_dof_idx)
    tip_state = f"{tip_param}{tip_node}"

    print("Kalman Filter configured:")
    print(f"  - Primary measurement: tip position ({tip_state})")
    print(f"  - Measurement noise std: {measurement_noise_std*1000:.1f}mm")
    print(f"  - State dimension: {n_states}")
    print(f"  - Time step: {dt}s")

    return HybridContinuousDiscreteKalman(A - B @ K, B, H, Q, R, P, x0, dt)


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
        n_states, amplitude=impulse_amplitude, duration=0.01
    )

    # Initialize random seed for repeatable noisy measurements
    np.random.seed(hash(case_name) % 2**32)

    def system_with_control_estimation(t, x):
        """System dynamics with control and estimation."""
        # External disturbance
        disturbance = impulse_func(t)

        if controller is None:
            # Uncontrolled case
            control_input = np.zeros(n_states)
        elif kalman_filter is None:
            # Perfect state feedback case
            reference = np.zeros_like(x)
            control_input = controller.compute_input(x, reference, t)
        else:
            # Kalman filter estimation case
            # Extract multiple position measurements (matching H matrix design)
            n_dofs = n_states // 2  # Half are positions, half are velocities
            n_measurements = min(5, n_dofs)
            measurement_indices = np.linspace(0, n_dofs - 1, n_measurements, dtype=int)

            # Create measurements for all monitored DOFs
            measurements = []
            for dof_idx in measurement_indices:
                true_position = x[dof_idx]
                noise = np.random.normal(0, measurement_noise_std)
                measurements.append(true_position + noise)
            tip_measurement = np.array(measurements)

            # Get control input from disturbance (for Kalman filter prediction)
            control_for_estimation = np.zeros_like(disturbance)

            # Estimate state using Kalman filter
            estimated_state = kalman_filter.estimate_states(
                tip_measurement, control_for_estimation, t
            )

            # Compute control input using estimated state
            reference = np.zeros_like(estimated_state)
            control_input = controller.compute_input(estimated_state, reference, t)

        # Combined input: disturbance + control
        total_input = disturbance + control_input

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

    return case_name, solution, computation_time, solver_stats


def create_multi_scenario_animation(solutions):
    """Create animation comparing multiple control scenarios."""
    print("Creating multi-scenario comparison animation...")

    # Extract beam shapes for all solutions
    props = get_material_properties()
    dx = props["length"]

    scenario_data = {}
    all_y_values = []

    for case_name, solution in solutions.items():
        x_coords, y_coords = extract_beam_shapes(solution, N_SEGMENTS, dx, linear=True)
        scenario_data[case_name] = (x_coords, y_coords)
        all_y_values.extend(y_coords.flatten())

    # Create animation with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))

    # Setup animation plot
    ax1.set_xlim(0.0, 1.6)
    y_min, y_max = np.min(all_y_values), np.max(all_y_values)
    y_pad = (y_max - y_min) * 0.1
    ax1.set_ylim(y_min - y_pad, y_max + y_pad)

    ax1.set_xlabel("Beam Length (m)")
    ax1.set_ylabel("Displacement (m)")
    ax1.set_title("Control and Estimation Scenarios Comparison")
    ax1.grid(True)

    # Create line objects for animation
    colors = {
        "Uncontrolled": "red",
        "LQR (Full State)": "blue",
        "LQR + Kalman Filter": "green",
        "LQR (Perfect Sensor)": "purple",
    }
    styles = {
        "Uncontrolled": "-",
        "LQR (Full State)": "-",
        "LQR + Kalman Filter": "--",
        "LQR (Perfect Sensor)": ":",
    }

    lines = {}
    for case_name in scenario_data.keys():
        color = colors.get(case_name, "black")
        style = styles.get(case_name, "-")
        (line,) = ax1.plot(
            [], [], color=color, linestyle=style, linewidth=2, label=case_name
        )
        lines[case_name] = line

    ax1.legend()

    # Setup tip displacement comparison
    tip_responses = {}

    for case_name, solution in solutions.items():
        y_array = np.array(solution.y) if isinstance(solution.y, list) else solution.y
        n_pos = len(y_array) // 2
        tip_displacement = y_array[n_pos - 2, :]  # Tip transverse displacement
        tip_responses[case_name] = tip_displacement

        color = colors.get(case_name, "black")
        style = styles.get(case_name, "-")
        ax2.plot(
            solution.t,
            tip_displacement,
            color=color,
            linestyle=style,
            label=case_name,
            linewidth=2,
        )

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Tip Displacement (m)")
    ax2.set_title("Tip Response Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add performance metrics
    performance_text = "Performance Metrics:\n"
    baseline_case = "Uncontrolled"
    if baseline_case in tip_responses:
        baseline_response = tip_responses[baseline_case]
        baseline_peak = np.max(np.abs(baseline_response))
        baseline_time = list(solutions.values())[0].t
        baseline_settling = calculate_settling_time(baseline_time, baseline_response)

        performance_text += f"{baseline_case}:\n"
        performance_text += f"  Peak: {baseline_peak:.4f}m\n"
        performance_text += f"  Settling: {baseline_settling:.3f}s\n\n"

        for case_name, tip_response in tip_responses.items():
            if case_name != baseline_case:
                peak = np.max(np.abs(tip_response))
                time_array = solutions[case_name].t
                settling = calculate_settling_time(time_array, tip_response)

                peak_reduction = (baseline_peak - peak) / baseline_peak * 100
                settling_improvement = (
                    (baseline_settling - settling) / baseline_settling * 100
                )

                performance_text += f"{case_name}:\n"
                performance_text += f"  Peak: {peak:.4f}m ({peak_reduction:+.1f}%)\n"
                performance_text += (
                    f"  Settling: {settling:.3f}s ({settling_improvement:+.1f}%)\n\n"
                )

    ax2.text(
        0.02,
        0.98,
        performance_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=9,
    )

    def animate(frame):
        """Animation function for multiple scenarios."""
        animated_objects = []
        for case_name, (x_coords, y_coords) in scenario_data.items():
            if frame < len(x_coords) and frame < len(y_coords):
                lines[case_name].set_data(x_coords[frame], y_coords[frame])
                animated_objects.append(lines[case_name])
        return animated_objects

    # Create animation
    min_frames = min(len(coords[0]) for coords in scenario_data.values())
    anim = FuncAnimation(
        fig, animate, frames=min_frames, interval=DT * 1000, blit=True, repeat=True
    )

    plt.tight_layout()
    return anim


def calculate_settling_time(time, response, tolerance=0.02):
    """Calculate settling time (time to stay within 2% of steady state)."""
    steady_state = response[-1]  # Assume final value is steady state

    # Find where response enters and stays within tolerance
    within_tolerance = (
        np.abs(response - steady_state) <= tolerance * np.abs(steady_state)
        if steady_state != 0
        else np.abs(response) <= tolerance
    )

    # Find last time it went outside tolerance
    if np.all(within_tolerance):
        return 0.0  # Always within tolerance

    last_outside = np.where(~within_tolerance)[0]
    if len(last_outside) == 0:
        return 0.0

    return time[last_outside[-1]] if last_outside[-1] + 1 < len(time) else time[-1]


def main(kalman_only=False):
    """Main function to demonstrate LQR control with and without Kalman filtering.

    Args:
        kalman_only: If True, only runs the Kalman filter simulation case
    """
    print("=" * 80)
    print("LQR CONTROL WITH KALMAN FILTER ESTIMATION COMPARISON")
    print("=" * 80)
    print("Comparing multiple control and estimation scenarios:")
    print("1. Uncontrolled system")
    print("2. LQR with full state feedback (ideal case)")
    print("3. LQR with Kalman filter estimation (realistic case)")
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

        # Design LQR controller and Kalman filter
        print("\nDesigning control and estimation systems...")
        controller, lqr_controller = design_lqr_controller(reference_beam)
        kalman_filter = design_kalman_filter(reference_beam, lqr_controller)

        # Define simulation scenarios
        measurement_noise_std = 0.002  # 0.2cm
        impulse_amplitude = 0.01

        if kalman_only:
            # Run only the Kalman filter case for troubleshooting
            simulation_tasks = [
                {
                    "case_name": "LQR + Kalman Filter",
                    "param_file": linear_file,
                    "controller": controller,
                    "kalman_filter": kalman_filter,
                    "impulse_amplitude": impulse_amplitude,
                    "measurement_noise_std": measurement_noise_std,
                }
            ]
        else:
            simulation_tasks = [
                {
                    "case_name": "Uncontrolled",
                    "param_file": linear_file,
                    "controller": None,
                    "kalman_filter": None,
                    "impulse_amplitude": impulse_amplitude,
                    "measurement_noise_std": measurement_noise_std,
                },
                {
                    "case_name": "LQR (Full State)",
                    "param_file": linear_file,
                    "controller": controller,
                    "kalman_filter": None,
                    "impulse_amplitude": impulse_amplitude,
                    "measurement_noise_std": measurement_noise_std,
                },
                {
                    "case_name": "LQR + Kalman Filter",
                    "param_file": linear_file,
                    "controller": controller,
                    "kalman_filter": kalman_filter,
                    "impulse_amplitude": impulse_amplitude,
                    "measurement_noise_std": measurement_noise_std,
                },
            ]

        print(f"\nStarting parallel simulation of {len(simulation_tasks)} scenarios...")
        start_time = time.time()

        # Run simulations sequentially
        results = []
        for task in simulation_tasks:
            result = simulate_control_scenario(task)
            results.append(result)

        total_time = time.time() - start_time
        print(f"\nAll simulations completed in {total_time:.2f} seconds")

        # Process results
        solutions = {}
        computation_times = {}
        solver_statistics = {}

        for case_name, solution, comp_time, solver_stats in results:
            solutions[case_name] = solution
            computation_times[case_name] = comp_time
            solver_statistics[case_name] = solver_stats

        # Print performance comparison table
        print_performance_table(computation_times, solver_statistics)

        # Print control performance comparison
        print("\n" + "=" * 80)
        print("CONTROL AND ESTIMATION PERFORMANCE ANALYSIS")
        print("=" * 80)

        uncontrolled_solution = solutions.get("Uncontrolled")
        lqr_solution = solutions.get("LQR (Full State)")
        kalman_solution = solutions.get("LQR + Kalman Filter")

        if all(
            sol is not None
            for sol in [uncontrolled_solution, lqr_solution, kalman_solution]
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
            tip_kalman = get_tip_response(kalman_solution)

            # Calculate performance metrics
            peak_uncontrolled = np.max(np.abs(tip_uncontrolled))
            peak_lqr = np.max(np.abs(tip_lqr))
            peak_kalman = np.max(np.abs(tip_kalman))

            settling_uncontrolled = calculate_settling_time(
                uncontrolled_solution.t, tip_uncontrolled
            )
            settling_lqr = calculate_settling_time(lqr_solution.t, tip_lqr)
            settling_kalman = calculate_settling_time(kalman_solution.t, tip_kalman)

            print("Peak Displacement:")
            print(f"  Uncontrolled:     {peak_uncontrolled:.4f}m")
            print(
                f"  LQR (Full State): {peak_lqr:.4f}m ({(peak_uncontrolled-peak_lqr)/peak_uncontrolled*100:+.1f}%)"
            )
            print(
                f"  LQR + Kalman:     {peak_kalman:.4f}m ({(peak_uncontrolled-peak_kalman)/peak_uncontrolled*100:+.1f}%)"
            )

            print("\nSettling Time:")
            print(f"  Uncontrolled:     {settling_uncontrolled:.3f}s")
            print(
                f"  LQR (Full State): {settling_lqr:.3f}s ({(settling_uncontrolled-settling_lqr)/settling_uncontrolled*100:+.1f}%)"
            )
            print(
                f"  LQR + Kalman:     {settling_kalman:.3f}s ({(settling_uncontrolled-settling_kalman)/settling_uncontrolled*100:+.1f}%)"
            )

            print("\nEstimation vs Perfect Feedback:")
            print(
                f"  Peak degradation:     {(peak_kalman-peak_lqr)/peak_lqr*100:+.1f}%"
            )
            print(
                f"  Settling degradation: {(settling_kalman-settling_lqr)/settling_lqr*100:+.1f}%"
            )

        # Create and display animation
        print("\nCreating comparison animation...")
        create_multi_scenario_animation(solutions)
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

    # Check for command line argument to run only Kalman filter case
    kalman_only = len(sys.argv) > 1 and sys.argv[1].lower() in [
        "kalman",
        "kalman-only",
        "ekf",
        "filter",
    ]
    main(kalman_only=True)
