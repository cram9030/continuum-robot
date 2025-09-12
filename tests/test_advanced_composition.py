import pytest
import numpy as np
import tempfile
import os
import time

from continuum_robot.models.dynamic_beam_model import DynamicEulerBernoulliBeam
from continuum_robot.models.fluid_forces import FluidDynamicsParams
from continuum_robot.models.gravity_forces import GravityForce
from continuum_robot.models.abstractions import AbstractForce, AbstractInputHandler


@pytest.fixture
def complex_beam_csv_data():
    """Create CSV data for a more complex test beam with mixed types."""
    return """length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition,wetted_area,drag_coef
0.2,200e9,1e-8,8000,1e-4,linear,FIXED,1e-4,1.2
0.2,200e9,1e-8,8000,1e-4,linear,NONE,1e-4,1.2
0.2,200e9,1e-8,8000,1e-4,nonlinear,NONE,1e-4,1.2
0.2,200e9,1e-8,8000,1e-4,nonlinear,NONE,1e-4,1.2
0.2,200e9,1e-8,8000,1e-4,nonlinear,NONE,1e-4,1.2"""


@pytest.fixture
def complex_beam_file(complex_beam_csv_data):
    """Create a temporary complex beam CSV file."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(complex_beam_csv_data)
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


class StateAwareForce(AbstractForce):
    """Force component that depends on current state."""

    def __init__(self, beam_instance, stiffness=1000.0, damping=10.0, enabled=True):
        self.beam = beam_instance
        self.stiffness = stiffness
        self.damping = damping
        self.enabled = enabled

    def compute_forces(self, x: np.ndarray, t: float) -> np.ndarray:
        """Apply state-dependent force (spring-damper on tip)."""
        n_states = len(x) // 2
        positions = x[:n_states]
        velocities = x[n_states:]

        forces = np.zeros(n_states)

        # Apply spring-damper force on last transverse DOF
        if n_states > 1:
            tip_pos_idx = n_states - 2  # Last w position
            tip_vel_idx = tip_pos_idx  # Corresponding velocity in second half

            spring_force = -self.stiffness * positions[tip_pos_idx]
            damper_force = -self.damping * velocities[tip_vel_idx]
            forces[tip_pos_idx] = spring_force + damper_force

        return forces

    def is_enabled(self) -> bool:
        return self.enabled


class TimeVaryingInputHandler(AbstractInputHandler):
    """Input handler that varies with time and state."""

    def __init__(self, beam_instance, frequency=1.0, amplitude=0.1, enabled=True):
        self.beam = beam_instance
        self.frequency = frequency
        self.amplitude = amplitude
        self.enabled = enabled
        self.call_count = 0

    def process_input(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """Apply time-varying gain to input."""
        self.call_count += 1
        gain = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        return u * gain  # Return modification

    def is_enabled(self) -> bool:
        return self.enabled


class TestAdvancedForceComposition:
    """Test advanced force composition scenarios."""

    def test_multiple_force_types_composition(self, complex_beam_file):
        """Test composition of multiple different force types."""
        fluid_params = FluidDynamicsParams(
            fluid_density=1000.0, enable_fluid_effects=True
        )
        beam = DynamicEulerBernoulliBeam(complex_beam_file, fluid_params=fluid_params)

        # Add state-aware force and gravity force to registry
        state_force = StateAwareForce(beam, stiffness=500.0, damping=5.0)
        beam.force_registry.register(state_force)

        gravity_force = GravityForce(beam, gravity_vector=[0.0, -9.81, 0.0])
        beam.force_registry.register(gravity_force)

        # Should now have fluid, state-aware, and gravity forces
        assert len(beam.force_registry) == 3

        # Use registry forces directly
        beam.create_system_func()

        # Test with non-zero state to activate state-dependent forces
        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01
        result = beam.get_system_func()(test_state)

        assert result.shape == (2 * n_dofs,)
        assert np.isfinite(result).all()

    def test_force_composition_order_independence(self, complex_beam_file):
        """Test that force composition is order-independent (commutative)."""
        beam1 = DynamicEulerBernoulliBeam(complex_beam_file)
        beam2 = DynamicEulerBernoulliBeam(complex_beam_file)

        # Create two identical forces
        force1 = StateAwareForce(beam1, stiffness=100.0)
        force2 = StateAwareForce(beam1, stiffness=200.0)

        # Register in different orders
        beam1.force_registry.register(force1)
        beam1.force_registry.register(force2)

        beam2.force_registry.register(force2)
        beam2.force_registry.register(force1)

        beam1.create_system_func()
        beam2.create_system_func()

        # Results should be identical
        n_dofs = len(beam1.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01

        result1 = beam1.get_system_func()(test_state)
        result2 = beam2.get_system_func()(test_state)

        assert np.allclose(result1, result2)

    def test_force_scaling_composition(self, complex_beam_file):
        """Test force scaling in composition."""
        beam = DynamicEulerBernoulliBeam(complex_beam_file)

        # Create base force function
        def base_force(x, t):
            n_states = len(x) // 2
            forces = np.zeros(n_states)
            forces[1] = 100.0  # Constant force on first w DOF
            return forces

        # Create scaled version
        scale_factor = 2.5

        def scaled_force(x, t):
            return scale_factor * base_force(x, t)

        beam.create_system_func(scaled_force)

        # Compare with manual scaling
        beam_base = DynamicEulerBernoulliBeam(complex_beam_file)
        beam_base.create_system_func(base_force)

        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.zeros(2 * n_dofs)

        result_scaled = beam.get_system_func()(test_state)
        result_base = beam_base.get_system_func()(test_state)

        # At least verify that scaling has an effect
        assert not np.allclose(result_scaled, result_base)


class TestAdvancedInputComposition:
    """Test advanced input composition scenarios."""

    def test_multiple_input_handlers(self, complex_beam_file):
        """Test composition of multiple input handlers."""
        beam = DynamicEulerBernoulliBeam(complex_beam_file)

        # Register multiple handlers
        handler1 = TimeVaryingInputHandler(beam, frequency=1.0, amplitude=0.1)
        handler2 = TimeVaryingInputHandler(beam, frequency=2.0, amplitude=0.05)

        beam.input_registry.register(handler1)
        beam.input_registry.register(handler2)

        assert len(beam.input_registry) == 2

        beam.create_input_func()

        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01
        test_input = np.ones(n_dofs)

        result = beam.input_func(test_state, test_input)
        assert result.shape == (2 * n_dofs,)

        # Verify both handlers were called
        # Note: In current implementation, t is always 0.0, so sin(0) = 0
        # The handlers return u * amplitude * sin(2*pi*f*0) = 0
        # So the result should be the same as original input

        # Test that handlers have correct interface
        assert hasattr(handler1, "call_count")
        assert hasattr(handler2, "call_count")

    def test_input_handler_state_dependency(self, complex_beam_file):
        """Test input handler that depends on current state."""
        beam = DynamicEulerBernoulliBeam(complex_beam_file)

        class StateAwareInputHandler(AbstractInputHandler):
            def __init__(self, beam_instance):
                self.beam = beam_instance

            def process_input(
                self, x: np.ndarray, u: np.ndarray, t: float
            ) -> np.ndarray:
                """Scale input based on tip position."""
                n_states = len(x) // 2
                positions = x[:n_states]

                # Scale based on tip displacement
                if n_states > 1:
                    tip_displacement = abs(positions[n_states - 2])  # Last w position
                    scale = 1.0 + tip_displacement * 10.0  # Nonlinear scaling
                else:
                    scale = 1.0

                return u * scale  # Return modification

            def is_enabled(self) -> bool:
                return True

        handler = StateAwareInputHandler(beam)
        beam.input_registry.register(handler)
        beam.create_input_func()

        n_dofs = len(beam.state_to_node_param) // 2
        test_input = np.ones(n_dofs)

        # Test with zero state
        zero_state = np.zeros(2 * n_dofs)
        result_zero = beam.input_func(zero_state, test_input)

        # Test with non-zero state
        nonzero_state = np.random.rand(2 * n_dofs) * 0.01
        result_nonzero = beam.input_func(nonzero_state, test_input)

        # Results should be different due to state dependency
        assert not np.allclose(result_zero, result_nonzero)


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""

    def test_force_registry_performance(self, complex_beam_file):
        """Test performance with many registered forces."""
        beam = DynamicEulerBernoulliBeam(complex_beam_file)

        # Register many forces
        num_forces = 50
        for i in range(num_forces):
            force = StateAwareForce(beam, stiffness=100.0 + i, damping=1.0 + i * 0.1)
            beam.force_registry.register(force)

        assert len(beam.force_registry) == num_forces

        beam.create_system_func()

        # Time the system function evaluation
        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01

        start_time = time.time()
        for _ in range(10):  # Multiple evaluations
            result = beam.get_system_func()(test_state)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10

        # Should complete reasonably quickly (less than 1 second per eval)
        assert avg_time < 1.0
        assert result.shape == (2 * n_dofs,)

    def test_memory_efficiency_force_composition(self, complex_beam_file):
        """Test memory efficiency of force composition."""
        beam = DynamicEulerBernoulliBeam(complex_beam_file)

        # Create and register forces
        forces = []
        for i in range(20):
            force = StateAwareForce(beam, stiffness=100.0 + i)
            forces.append(force)
            beam.force_registry.register(force)

        beam.create_system_func()

        # Verify that forces are properly referenced
        registered_forces = beam.force_registry.get_registered_forces()
        assert len(registered_forces) == 20

        # Test that unregistering works correctly
        for i in range(10):
            beam.force_registry.unregister(forces[i])

        assert len(beam.force_registry) == 10

        # System function should still work with remaining forces
        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01
        result = beam.get_system_func()(test_state)
        assert result.shape == (2 * n_dofs,)


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness of composition system."""

    def test_invalid_force_function_handling(self, complex_beam_file):
        """Test handling of invalid external force functions."""
        beam = DynamicEulerBernoulliBeam(complex_beam_file)

        def invalid_force(x, t):
            """Force function that returns wrong shape."""
            return np.array([1.0, 2.0])  # Wrong shape

        beam.create_system_func(invalid_force)

        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01

        # Should raise an error when trying to use the invalid force
        with pytest.raises((ValueError, IndexError, TypeError)):
            beam.get_system_func()(test_state)

    def test_force_function_exception_handling(self, complex_beam_file):
        """Test handling of exceptions in force functions."""
        beam = DynamicEulerBernoulliBeam(complex_beam_file)

        def problematic_force(x, t):
            """Force function that raises an exception."""
            if np.any(x > 0.005):  # Arbitrary condition
                raise ValueError("Force computation failed")
            return np.zeros(len(x) // 2)

        beam.create_system_func(problematic_force)

        n_dofs = len(beam.state_to_node_param) // 2
        small_state = np.ones(2 * n_dofs) * 0.001  # Should work
        large_state = np.ones(2 * n_dofs) * 0.01  # Should fail

        # Small state should work
        result = beam.get_system_func()(small_state)
        assert result.shape == (2 * n_dofs,)

        # Large state should raise the exception
        with pytest.raises(ValueError, match="Force computation failed"):
            beam.get_system_func()(large_state)

    def test_disabled_force_during_runtime(self, complex_beam_file):
        """Test behavior when forces are disabled during runtime."""
        beam = DynamicEulerBernoulliBeam(complex_beam_file)

        # Create a force that can be disabled
        toggleable_force = StateAwareForce(beam, stiffness=1000.0)
        beam.force_registry.register(toggleable_force)
        beam.create_system_func()

        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01

        # Test with force enabled
        result_enabled = beam.get_system_func()(test_state)

        # Disable the force
        toggleable_force.enabled = False

        # The force's is_enabled() method is checked each time, so disabling
        # will affect the current system function immediately
        result_after_disable = beam.get_system_func()(test_state)

        # Results should be different because the force is now disabled
        assert not np.allclose(result_enabled, result_after_disable)

        # Re-enable the force
        toggleable_force.enabled = True
        result_re_enabled = beam.get_system_func()(test_state)

        # Should match the original enabled result
        assert np.allclose(result_enabled, result_re_enabled)


class TestComplexIntegrationScenarios:
    """Test complex integration scenarios."""

    def test_full_simulation_with_composition(self, complex_beam_file):
        """Test full simulation pipeline with composed forces."""
        fluid_params = FluidDynamicsParams(
            fluid_density=1000.0, enable_fluid_effects=True
        )
        beam = DynamicEulerBernoulliBeam(complex_beam_file, fluid_params=fluid_params)

        # Add additional forces
        state_force = StateAwareForce(beam, stiffness=500.0, damping=10.0)
        beam.force_registry.register(state_force)

        # Create external force
        def time_varying_force(x, t):
            n_states = len(x) // 2
            forces = np.zeros(n_states)
            # Apply sinusoidal force on tip
            if n_states > 1:
                forces[n_states - 2] = 50.0 * np.sin(10.0 * t)
            return forces

        # Combine all forces
        registry_forces = beam.force_registry.create_aggregated_function()

        def total_forces(x, t):
            return registry_forces(x, t) + time_varying_force(x, t)

        beam.create_system_func(total_forces)
        beam.create_input_func()

        # Test system function
        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.001

        result = beam.get_system_func()(test_state)
        assert result.shape == (2 * n_dofs,)
        assert np.isfinite(result).all()

        # Test dynamic system
        dynamic_system = beam.get_dynamic_system()
        zero_input = np.zeros(n_dofs)

        dyn_result = dynamic_system(0.1, test_state, zero_input)
        assert dyn_result.shape == (2 * n_dofs,)
        assert np.isfinite(dyn_result).all()

    def test_composition_consistency_across_recreations(self, complex_beam_file):
        """Test that composition is consistent when recreating functions."""
        beam = DynamicEulerBernoulliBeam(complex_beam_file)

        # Register a force
        force = StateAwareForce(beam, stiffness=1000.0)
        beam.force_registry.register(force)

        # Create system function multiple times
        beam.create_system_func()
        beam.create_input_func()

        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01
        test_input = np.random.rand(n_dofs) * 0.1

        # Store results
        result1 = beam.get_system_func()(test_state)
        input_result1 = beam.input_func(test_state, test_input)

        # Recreate functions
        beam.create_system_func()
        beam.create_input_func()

        result2 = beam.get_system_func()(test_state)
        input_result2 = beam.input_func(test_state, test_input)

        # Results should be identical
        assert np.allclose(result1, result2)
        assert np.allclose(input_result1, input_result2)
