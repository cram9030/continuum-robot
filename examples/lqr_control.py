"""
LQR Control Example for Continuum Robot Beams with Kalman Filter Estimation.
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
from control_design.linear_quadratic_regulator import LinearQuadraticRegulator
from continuum_robot.control.full_state_linear import FullStateLinear
from continuum_robot.estimator.kalman_filter import DiscreteKalman
from continuum_robot.utils.discretization import continuous_to_discrete_zoh

from example_utilities import (
    create_beam_parameters,
    extract_beam_shapes,
    cleanup_temp_files,
    print_performance_table,
    get_material_properties,
    DT,
    N_SEGMENTS,
)

T_FINAL = 1


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
        DiscreteKalman: Configured discrete Kalman filter for tip position estimation
    """
    print("Designing Kalman Filter for tip position estimation...")

    # Get continuous-time system matrices from LQR controller
    A_cont = lqr_controller.get_A()
    B_cont = lqr_controller.get_B()
    K = lqr_controller.get_K()

    n_states = A_cont.shape[0]
    n_dofs = n_states // 2

    # Time step for discretization
    dt = 0.001  # 1ms for discrete updates

    # Discretize closed-loop system matrices using zero-order hold
    # Closed-loop dynamics: A_cl = A - B*K
    A_cl_cont = A_cont - B_cont @ K
    A_discrete, B_discrete = continuous_to_discrete_zoh(A_cl_cont, B_cont, dt)

    # Observation matrix C
    n_measurements = 4

    C = np.zeros((n_measurements, n_states))
    C[0, 17] = 1
    C[1, :] = A_cont[17, :]
    C[2, :] = A_cont[-3, :]
    C[3, :] = A_cont[-2, :]

    # Measurement noise covariance for multiple measurements
    R = measurement_noise_std**2 * np.eye(n_measurements)

    # Process noise covariance Q (discrete-time)
    # Discretize continuous-time process noise
    Q_cont = 1e-3 * np.eye(n_states)
    Q_discrete = Q_cont * dt  # First-order discretization

    # Initial error covariance P
    # Start with moderate uncertainty
    P = 1e-1 * np.eye(n_states)

    # Initial state estimate (zero)
    x0 = np.zeros(n_states)

    # Get tip state description (last transverse displacement)
    tip_dof_idx = n_dofs - 2  # Tip transverse displacement w at last node
    tip_param, tip_node = beam.beam_model.get_dof_to_node_param(tip_dof_idx)
    tip_state = f"{tip_param}{tip_node}"

    print("Kalman Filter configured:")
    print("  - Type: Discrete-time Kalman Filter")  # noqa: F541
    print(f"  - Primary measurement: tip position ({tip_state})")
    print(f"  - Measurement noise std: {measurement_noise_std*1000:.1f}mm")
    print(f"  - State dimension: {n_states}")
    print(f"  - Sampling time: {dt}s ({1/dt:.0f} Hz)")

    return DiscreteKalman(A_discrete, B_discrete, C, Q_discrete, R, P, x0)


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

    # For discrete Kalman filter: log estimates at discrete intervals
    dt_kalman = 0.001  # 1ms sampling time for discrete Kalman filter
    estimate_log = {}  # Dictionary mapping time -> estimated_state

    def get_estimate_at_time(t):
        """Get the most recent estimate at or before time t."""
        if not estimate_log:
            return np.zeros(n_full_states)

        # Find the most recent logged estimate at or before t
        available_times = [time for time in estimate_log.keys() if time <= t]
        if not available_times:
            return np.zeros(n_full_states)

        most_recent_time = max(available_times)
        return estimate_log[most_recent_time]

    def system_with_control_estimation(t, x):
        """System dynamics with control and estimation."""
        # External disturbance
        disturbance = impulse_func(t)

        if controller is None:
            # Uncontrolled case
            control_input = np.zeros(n_states)
        elif kalman_filter is None:
            # Perfect state feedback case
        if controller is None:
            # Uncontrolled case
            control_input = np.zeros(n_states)
        elif kalman_filter is None:
            # Perfect state feedback case
            reference = np.zeros_like(x)
            control_input = controller.compute_input(x, reference, t)
        else:
            # Kalman filter estimation case (discrete-time)
            # Determine if we should run filter update at this time
            # Round to nearest dt_kalman to determine discrete sample times
            sample_index = int(np.round(t / dt_kalman))
            sample_time = sample_index * dt_kalman

            # Only update if this exact sample time hasn't been logged yet
            # and we're close enough to the sample time (within tolerance)
            if (
                sample_time not in estimate_log
                and abs(t - sample_time) < dt_kalman * 0.1
            ):
                # Time for new measurement and filter update
                n_measurements = 4

                # Create measurements for all monitored DOFs
                noise = np.random.normal(
                    0, measurement_noise_std, size=(n_measurements,)
                )
                y = kalman_filter.C @ x + noise

                # Get control input for estimation (zero for disturbance-only system)
                control_for_estimation = np.zeros(n_states)

                # Update estimate using Kalman filter
                estimated_state = kalman_filter.estimate_states(
                    y, control_for_estimation, sample_time
                )

                # Log the estimate at this sample time
                estimate_log[sample_time] = estimated_state.copy()

            # Use most recent logged estimate for control
            estimated_state = get_estimate_at_time(t)

            # Compute control input using estimated state
            reference = np.zeros_like(estimated_state)
            control_input = controller.compute_input(x, reference, t)

        # Combined input: disturbance + control
        total_input = disturbance + control_input

        # System dynamics
        return beam.get_dynamic_system()(t, x, total_input)

    # Solve system
    t_span = (0, T_FINAL)
    t_eval = np.arange(0, T_FINAL, DT)

    solution = solve_ivp(
        system_with_control_estimation,
        system_with_control_estimation,
        t_span,
        x0,
        method="LSODA",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
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

    return case_name, solution, computation_time, solver_stats, estimate_log


def create_multi_scenario_animation(solutions, estimate_logs):  # noqa: C901
    """Create animation comparing multiple control scenarios.

    Args:
        solutions: Dictionary mapping case_name -> solution object
        estimate_logs: Dictionary mapping case_name -> estimate_log dictionary
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
            # Create pseudo-solution object from estimate_log for beam shape extraction
            estimate_log = estimate_logs[case_name]
            times = sorted(estimate_log.keys())

            # Build state array from logged estimates
            n_states = len(list(estimate_log.values())[0]) if estimate_log else 0
            if n_states > 0:
                est_states = np.array([estimate_log[t] for t in times]).T

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

    # Create animation with 2 subplots
    # Create animation with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))

    # Setup animation plot
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
        "LQR + Kalman Filter": "-",
        "LQR (Perfect Sensor)": ":",
    }

    lines = {}
    estimate_lines = {}

    for case_name in scenario_data.keys():
        color = colors.get(case_name, "black")
        style = styles.get(case_name, "-")

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

        color = colors.get(case_name, "black")
        style = styles.get(case_name, "-")

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
    min_frames = min(len(coords[0]) for coords in scenario_data.values())
    anim = FuncAnimation(
        fig, animate, frames=min_frames, interval=DT * 1000, blit=True, repeat=True
        fig, animate, frames=min_frames, interval=DT * 1000, blit=True, repeat=True
    )

    plt.tight_layout()
    return anim


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
    print("-" * 80)

    # Create beam parameter files
    # Create beam parameter files
    linear_file, _, _ = create_beam_parameters()

    try:
        # Create reference beam for controller design
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

        for case_name, solution, comp_time, solver_stats, estimate_log in results:
            solutions[case_name] = solution
            computation_times[case_name] = comp_time
            solver_statistics[case_name] = solver_stats
            estimate_logs[case_name] = estimate_log

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

            print("Peak Displacement:")
            print(f"  Uncontrolled:     {peak_uncontrolled:.3f}m")
            print(
                f"  LQR (Full State): {peak_lqr:.3f}m ({(peak_uncontrolled-peak_lqr)/peak_uncontrolled*100:+.1f}%)"
            )
            print(
                f"  LQR + Kalman:     {peak_kalman:.3f}m ({(peak_uncontrolled-peak_kalman)/peak_uncontrolled*100:+.1f}%)"
            )

            print("\nEstimation vs Perfect Feedback:")
            print(
                f"  Peak degradation:     {(peak_kalman-peak_lqr)/peak_lqr*100:+.1f}%"
            )

        # Create and display animation
        print("\nCreating comparison animation...")
        _ = create_multi_scenario_animation(solutions, estimate_logs)  # noqa: F841
        plt.show()

        print("\nSimulation completed successfully!")
        print("Close the animation window to exit.")

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback

        traceback.print_exc()
        traceback.print_exc()
    finally:
        # Cleanup
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
    main(kalman_only=False)
