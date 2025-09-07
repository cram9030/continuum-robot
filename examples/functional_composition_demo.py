#!/usr/bin/env python3
"""
Demonstration of the new functional composition approach in DynamicEulerBernoulliBeam.

This example shows how to:
1. Use the default registry-based approach
2. Create custom force functions and pass them externally
3. Combine registry forces with external forces
"""

import numpy as np
import tempfile
import os
from pathlib import Path
from continuum_robot.models.dynamic_beam_model import DynamicEulerBernoulliBeam
from continuum_robot.models.fluid_forces import FluidDynamicsParams
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_demo_beam_file():
    """Create a demo beam CSV file for testing."""
    demo_data = """length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition,wetted_area,drag_coef
0.25,200e9,1e-8,8000,1e-4,linear,FIXED,1e-4,1.2
0.25,200e9,1e-8,8000,1e-4,linear,NONE,1e-4,1.2
0.25,200e9,1e-8,8000,1e-4,linear,NONE,1e-4,1.2
0.25,200e9,1e-8,8000,1e-4,linear,NONE,1e-4,1.2"""

    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write(demo_data)
    return path


def custom_gravity_force(x, t):
    """Custom gravity force function."""
    n_states = len(x) // 2
    gravity_forces = np.zeros(n_states)

    # Apply downward gravity force on transverse DOFs (w)
    for i in range(1, n_states, 3):  # w DOFs are at indices 1, 4, 7, ...
        gravity_forces[i] = -9.81 * 8000 * 1e-4 * 0.25  # rho * A * L * g

    return gravity_forces


def custom_spring_force(k=1000):
    """Factory for custom spring force at the tip."""

    def spring_force(x, t):
        n_states = len(x) // 2
        positions = x[:n_states]
        spring_forces = np.zeros(n_states)

        # Apply spring force at tip (last w position)
        if n_states > 1:
            tip_w_idx = n_states - 2  # Last w DOF
            spring_forces[tip_w_idx] = -k * positions[tip_w_idx]

        return spring_forces

    return spring_force


def main():
    """Demonstrate functional composition approaches."""
    beam_file = create_demo_beam_file()

    try:
        print("=== Functional Composition Demo ===\n")

        # 1. Registry-based approach (default)
        print("1. Registry-based approach (with fluid drag):")
        fluid_params = FluidDynamicsParams(
            fluid_density=1000.0, enable_fluid_effects=True
        )
        beam1 = DynamicEulerBernoulliBeam(beam_file, fluid_params=fluid_params)

        print(f"   Force registry has {len(beam1.force_registry)} registered forces")
        beam1.create_system_func()  # Uses default registry forces
        print("   System function created with registry forces")

        # 2. External custom force function
        print("\n2. External custom force function:")
        beam2 = DynamicEulerBernoulliBeam(beam_file)  # No fluid forces

        # Create custom combined force function
        def custom_forces(x, t):
            gravity = custom_gravity_force(x, t)
            spring = custom_spring_force(k=500)(x, t)
            return gravity + spring

        beam2.create_system_func(custom_forces)
        print(
            "   System function created with custom external forces (gravity + spring)"
        )

        # 3. Hybrid approach: registry + external forces
        print("\n3. Hybrid approach (registry + external):")
        beam3 = DynamicEulerBernoulliBeam(beam_file, fluid_params=fluid_params)

        # Get registry forces and combine with external
        registry_forces = beam3.force_registry.create_aggregated_function()

        def combined_forces(x, t):
            # Get forces from registry (fluid drag)
            registry_contrib = registry_forces(x, t)
            # Add external forces
            external_contrib = custom_gravity_force(x, t)
            return registry_contrib + external_contrib

        beam3.create_system_func(combined_forces)
        print(
            "   System function created with registry (fluid) + external (gravity) forces"
        )

        # 4. Dynamic force registration
        print("\n4. Dynamic force registration:")
        beam4 = DynamicEulerBernoulliBeam(beam_file)

        # Manually register a fluid force later
        if beam4.fluid_params.enable_fluid_effects:
            from continuum_robot.models.fluid_forces import FluidDragForce

            custom_fluid = FluidDragForce(beam4)
            beam4.force_registry.register(custom_fluid)

        print(
            f"   Force registry now has {len(beam4.force_registry)} registered forces"
        )
        beam4.create_system_func()  # Uses updated registry
        print("   System function created with dynamically registered forces")

        # 5. Test system functions
        print("\n5. Testing system functions:")
        n_dofs = len(beam1.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01  # Small random state

        # Test all beams
        for i, beam in enumerate([beam1, beam2, beam3, beam4], 1):
            beam.create_input_func()  # Also create input function
            system_result = beam.get_system_func()(test_state)
            print(
                f"   Beam {i} system evaluation: shape={system_result.shape}, norm={np.linalg.norm(system_result):.6f}"
            )

        print("\n=== Demo completed successfully! ===")

    finally:
        # Cleanup
        os.unlink(beam_file)


if __name__ == "__main__":
    main()
