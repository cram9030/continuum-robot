"""
LQR Control Example for Continuum Robot Beams.

This example demonstrates Linear Quadratic Regulator (LQR) control applied to
continuum robot beams. It compares controlled vs uncontrolled beam response
to impulse disturbances and creates animations showing the effectiveness of
the LQR controller.

The LQR controller is designed to stabilize the beam and reduce oscillations
after disturbances while maintaining optimal control effort.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

from continuum_robot.models.dynamic_beam_model import DynamicEulerBernoulliBeam
from continuum_robot.models.force_params import ForceParams
from continuum_robot.control.linear_quadratic_regulator import LinearQuadraticRegulator
from continuum_robot.control.full_state_linear import FullStateLinear

from example_utilities import (
    create_beam_parameters,
    extract_beam_shapes,
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

    return FullStateLinear(gain_matrix)


def simulate_system(beam, controller, impulse_func, case_name):
    """Simulate beam with or without LQR control."""
    print(f"Simulating {case_name}...")

    # Get system dimensions
    n_states = beam.beam_model.M.shape[0]
    x0 = np.zeros(2 * n_states)

    def system_with_inputs(t, x):
        """Combined system with control and disturbances."""
        # External disturbance
        disturbance = impulse_func(t)

        # LQR control input (assuming zero reference for regulation)
        if controller:
            reference = np.zeros_like(x)
            control_input = controller.compute_input(x, reference, t)
        else:
            control_input = np.zeros(n_states)

        # Combined input: disturbance + control
        total_input = disturbance + control_input

        # System dynamics
        return beam.get_dynamic_system()(t, x, total_input)

    # Solve system
    t_span = (0, T_FINAL)
    t_eval = np.arange(0, T_FINAL, DT)

    solution = solve_ivp(
        system_with_inputs,
        t_span,
        x0,
        method="LSODA",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )

    print(f"  - Simulation completed: {solution.message}")
    print(f"  - Function evaluations: {solution.nfev}")

    return solution


def create_comparison_animation(sol_uncontrolled, sol_controlled):
    """Create animation comparing controlled vs uncontrolled response."""
    print("Creating comparison animation...")

    # Extract beam shapes
    from example_utilities import get_material_properties

    props = get_material_properties()
    dx = props["length"]

    x_unc, y_unc = extract_beam_shapes(sol_uncontrolled, N_SEGMENTS, dx, linear=True)
    x_ctrl, y_ctrl = extract_beam_shapes(sol_controlled, N_SEGMENTS, dx, linear=True)

    # Create animation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))

    # Animation setup
    ax1.set_xlim(0.0, 1.6)
    (line_unc,) = ax1.plot([], [], "r-", linewidth=2, label="Uncontrolled")
    (line_ctrl,) = ax1.plot([], [], "b-", linewidth=2, label="Controlled")

    # Find global y-limits
    y_values = np.concatenate([y_unc.flatten(), y_ctrl.flatten()])
    y_min, y_max = np.min(y_values), np.max(y_values)
    y_pad = (y_max - y_min) * 0.1
    ax1.set_ylim(y_min - y_pad, y_max + y_pad)

    ax1.set_xlabel("Beam Length (m)")
    ax1.set_ylabel("Displacement (m)")
    ax1.set_title("Controlled vs Uncontrolled Beam Response")
    ax1.grid(True)
    ax1.legend()

    # Tip displacement comparison
    n_pos = len(sol_uncontrolled.y) // 2
    tip_unc = sol_uncontrolled.y[n_pos - 2, :]  # Tip transverse displacement
    tip_ctrl = sol_controlled.y[n_pos - 2, :]

    ax2.plot(sol_uncontrolled.t, tip_unc, "r-", label="Uncontrolled", linewidth=2)
    ax2.plot(sol_controlled.t, tip_ctrl, "b-", label="Controlled", linewidth=2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Tip Displacement (m)")
    ax2.set_title("Tip Response Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add performance metrics
    settling_time_unc = calculate_settling_time(sol_uncontrolled.t, tip_unc)
    settling_time_ctrl = calculate_settling_time(sol_controlled.t, tip_ctrl)
    max_displacement_unc = np.max(np.abs(tip_unc))
    max_displacement_ctrl = np.max(np.abs(tip_ctrl))

    ax2.text(
        0.02,
        0.98,
        f"Settling Time:\nUncontrolled: {settling_time_unc:.3f}s\nControlled: {settling_time_ctrl:.3f}s\n\n"
        f"Peak Displacement:\nUncontrolled: {max_displacement_unc:.4f}m\nControlled: {max_displacement_ctrl:.4f}m",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    def animate(frame):
        """Animation function."""
        if frame < len(sol_uncontrolled.t):
            line_unc.set_data(x_unc[frame], y_unc[frame])
            line_ctrl.set_data(x_ctrl[frame], y_ctrl[frame])
        return line_unc, line_ctrl

    # Create animation
    n_frames = min(len(x_unc), len(y_unc), len(x_ctrl), len(y_ctrl))
    anim = FuncAnimation(
        fig, animate, frames=n_frames, interval=DT * 1000, blit=True, repeat=True
    )

    plt.tight_layout()
    plt.show()

    # Print performance summary
    print("\n" + "=" * 60)
    print("CONTROL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(
        f"Peak Displacement Reduction: {(max_displacement_unc - max_displacement_ctrl) / max_displacement_unc * 100:.1f}%"
    )
    print(
        f"Settling Time Improvement: {(settling_time_unc - settling_time_ctrl) / settling_time_unc * 100:.1f}%"
    )
    print(f"Uncontrolled settling time: {settling_time_unc:.3f}s")
    print(f"Controlled settling time: {settling_time_ctrl:.3f}s")

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


def main():
    """Main function to demonstrate LQR control."""
    print("=" * 60)
    print("LQR CONTROL DEMONSTRATION")
    print("=" * 60)
    print("Comparing controlled vs uncontrolled continuum robot response")
    print(f"System: {N_SEGMENTS} segments, {T_FINAL}s simulation")
    print("-" * 60)

    # Create beam parameter files using existing utilities
    linear_file, _, _ = create_beam_parameters()

    try:
        # Create single beam model for both controller design and simulation
        # Use gravity and fluid effects for realistic simulation
        force_params = ForceParams(
            enable_gravity_effects=True, enable_fluid_effects=False
        )
        beam = DynamicEulerBernoulliBeam(linear_file, force_params)

        # Create system functions
        beam.create_system_func()
        beam.create_input_func()
        n_dofs = beam.beam_model.M.shape[0]

        # Design LQR controller using the same beam
        controller = design_lqr_controller(beam)

        # Create impulse disturbance
        impulse_func = create_impulse_function(n_dofs, amplitude=10.0, duration=0.01)

        # Simulate uncontrolled system
        sol_uncontrolled = simulate_system(
            beam, None, impulse_func, "Uncontrolled System"
        )

        # Simulate controlled system
        sol_controlled = simulate_system(
            beam, controller, impulse_func, "LQR Controlled System"
        )

        # Create comparison animation
        create_comparison_animation(sol_uncontrolled, sol_controlled)

        print("\nSimulation completed successfully!")
        print("Close the animation window to exit.")

        # Keep the plot open
        plt.show()

    finally:
        # Cleanup using existing utilities
        from example_utilities import cleanup_temp_files

        cleanup_temp_files(linear_file)


if __name__ == "__main__":
    main()
