import pytest
import numpy as np
import tempfile
import os
from scipy.integrate import solve_ivp

from continuum_robot.models.dynamic_beam_model import DynamicEulerBernoulliBeam
from continuum_robot.models.fluid_forces import FluidDynamicsParams, FluidDragForce
from continuum_robot.models.force_registry import ForceRegistry, InputRegistry
from continuum_robot.models.abstractions import AbstractForce, AbstractInputHandler


@pytest.fixture
def beam_csv_data():
    """Create CSV data for a test beam."""
    return """length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition,wetted_area,drag_coef
0.25,200e9,1e-8,8000,1e-4,linear,FIXED,1e-4,1.2
0.25,200e9,1e-8,8000,1e-4,linear,NONE,1e-4,1.2
0.25,200e9,1e-8,8000,1e-4,linear,NONE,1e-4,1.2
0.25,200e9,1e-8,8000,1e-4,linear,NONE,1e-4,1.2"""


@pytest.fixture
def beam_file(beam_csv_data):
    """Create a temporary beam CSV file."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(beam_csv_data)
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


class MockForce(AbstractForce):
    """Mock force for testing purposes."""

    def __init__(self, beam_instance, force_magnitude=100.0, enabled=True):
        self.beam = beam_instance
        self.force_magnitude = force_magnitude
        self.enabled = enabled

    def compute_forces(self, x: np.ndarray, t: float) -> np.ndarray:
        """Apply constant force on first transverse DOF."""
        n_states = len(x) // 2
        forces = np.zeros(n_states)
        if n_states > 1:
            forces[1] = self.force_magnitude  # First w DOF
        return forces

    def is_enabled(self) -> bool:
        return self.enabled


class MockInputHandler(AbstractInputHandler):
    """Mock input handler for testing purposes."""

    def __init__(self, beam_instance, gain=0.1, enabled=True):
        self.beam = beam_instance
        self.gain = gain
        self.enabled = enabled

    def process_input(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """Apply gain to input (returns modification, not full processed input)."""
        return u * self.gain  # Return modification to be added

    def is_enabled(self) -> bool:
        return self.enabled


def custom_gravity_force(x, t):
    """Custom gravity force function for testing."""
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


class TestRegistryBasedForces:
    """Test registry-based force functionality."""

    def test_default_registry_with_fluid_forces(self, beam_file):
        """Test default registry behavior with fluid forces enabled."""
        fluid_params = FluidDynamicsParams(
            fluid_density=1000.0, enable_fluid_effects=True
        )
        beam = DynamicEulerBernoulliBeam(beam_file, fluid_params=fluid_params)

        # Check that fluid force was auto-registered
        assert len(beam.force_registry) == 1
        forces = beam.force_registry.get_registered_forces()
        assert isinstance(forces[0], FluidDragForce)
        assert forces[0].is_enabled()

        # Create system function using registry
        beam.create_system_func()
        assert beam.system_func is not None

        # Test system function evaluation
        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01
        result = beam.get_system_func()(test_state)
        assert result.shape == (2 * n_dofs,)
        assert not np.isnan(result).any()

    def test_default_registry_without_fluid_forces(self, beam_file):
        """Test default registry behavior without fluid forces."""
        beam = DynamicEulerBernoulliBeam(beam_file)  # No fluid params

        # Check that no forces were auto-registered
        assert len(beam.force_registry) == 0

        # Create system function using empty registry
        beam.create_system_func()
        assert beam.system_func is not None

        # Test system function evaluation
        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01
        result = beam.get_system_func()(test_state)
        assert result.shape == (2 * n_dofs,)


class TestExternalCustomForces:
    """Test external custom force functions."""

    def test_external_force_function(self, beam_file):
        """Test system creation with external custom force function."""
        beam = DynamicEulerBernoulliBeam(beam_file)

        # Create custom combined force function
        def custom_forces(x, t):
            gravity = custom_gravity_force(x, t)
            spring = custom_spring_force(k=500)(x, t)
            return gravity + spring

        # Create system with external forces
        beam.create_system_func(custom_forces)
        assert beam.system_func is not None

        # Test system function evaluation
        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01
        result = beam.get_system_func()(test_state)
        assert result.shape == (2 * n_dofs,)

        # Verify forces are actually being applied
        # Test with non-zero state to see spring effect
        tip_displacement_state = np.zeros(2 * n_dofs)
        if n_dofs > 1:
            tip_displacement_state[n_dofs - 2] = 0.01  # Displace tip
        result_with_displacement = beam.get_system_func()(tip_displacement_state)

        # Should be different from zero state due to spring force
        zero_result = beam.get_system_func()(np.zeros(2 * n_dofs))
        assert not np.allclose(result_with_displacement, zero_result)

    def test_time_dependent_force(self, beam_file):
        """Test time-dependent external force."""
        beam = DynamicEulerBernoulliBeam(beam_file)

        def time_varying_force(x, t):
            n_states = len(x) // 2
            forces = np.zeros(n_states)
            # Sinusoidal force on first transverse DOF
            if n_states > 1:
                forces[1] = 100.0 * np.sin(2 * np.pi * t)
            return forces

        beam.create_system_func(time_varying_force)

        # The force function should be stored and callable
        # Note: In current implementation, time is always passed as 0.0 to forces_func
        # This is a limitation that could be improved in future versions
        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.zeros(2 * n_dofs)
        result = beam.get_system_func()(test_state)
        assert result.shape == (2 * n_dofs,)


class TestHybridApproach:
    """Test hybrid approach combining registry and external forces."""

    def test_registry_plus_external_forces(self, beam_file):
        """Test combining registry forces with external forces."""
        fluid_params = FluidDynamicsParams(
            fluid_density=1000.0, enable_fluid_effects=True
        )
        beam = DynamicEulerBernoulliBeam(beam_file, fluid_params=fluid_params)

        # Get registry forces function
        registry_forces = beam.force_registry.create_aggregated_function()

        def combined_forces(x, t):
            # Get forces from registry (fluid drag)
            registry_contrib = registry_forces(x, t)
            # Add external forces (gravity)
            external_contrib = custom_gravity_force(x, t)
            return registry_contrib + external_contrib

        beam.create_system_func(combined_forces)

        # Test system function evaluation
        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01
        result = beam.get_system_func()(test_state)
        assert result.shape == (2 * n_dofs,)

        # Compare with registry-only forces
        beam_registry_only = DynamicEulerBernoulliBeam(
            beam_file, fluid_params=fluid_params
        )
        beam_registry_only.create_system_func()  # Uses registry only
        result_registry_only = beam_registry_only.get_system_func()(test_state)

        # Combined forces should be different from registry-only
        # (assuming gravity contributes something non-zero)
        assert not np.allclose(result, result_registry_only, rtol=1e-10)


class TestDynamicForceRegistration:
    """Test dynamic force registration."""

    def test_manual_force_registration(self, beam_file):
        """Test manually registering forces after initialization."""
        beam = DynamicEulerBernoulliBeam(beam_file)

        # Initially no forces
        assert len(beam.force_registry) == 0

        # Manually register a mock force
        mock_force = MockForce(beam, force_magnitude=200.0)
        beam.force_registry.register(mock_force)

        assert len(beam.force_registry) == 1
        assert mock_force in beam.force_registry

        # Create system function using updated registry
        beam.create_system_func()

        # Test that the mock force is applied
        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.zeros(2 * n_dofs)
        result = beam.get_system_func()(test_state)

        # Should see effect of mock force on accelerations (second half of state derivative)
        accelerations = result[n_dofs:]
        assert not np.allclose(accelerations, np.zeros_like(accelerations))

    def test_force_unregistration(self, beam_file):
        """Test unregistering forces."""
        beam = DynamicEulerBernoulliBeam(beam_file)

        # Register forces
        mock_force1 = MockForce(beam, force_magnitude=100.0)
        mock_force2 = MockForce(beam, force_magnitude=200.0)
        beam.force_registry.register(mock_force1)
        beam.force_registry.register(mock_force2)

        assert len(beam.force_registry) == 2

        # Unregister one force
        success = beam.force_registry.unregister(mock_force1)
        assert success
        assert len(beam.force_registry) == 1
        assert mock_force1 not in beam.force_registry
        assert mock_force2 in beam.force_registry

        # Try to unregister non-existent force
        success = beam.force_registry.unregister(mock_force1)
        assert not success

    def test_force_registry_clear(self, beam_file):
        """Test clearing all forces from registry."""
        fluid_params = FluidDynamicsParams(
            fluid_density=1000.0, enable_fluid_effects=True
        )
        beam = DynamicEulerBernoulliBeam(beam_file, fluid_params=fluid_params)

        # Should have auto-registered fluid force
        assert len(beam.force_registry) == 1

        # Add another force
        mock_force = MockForce(beam)
        beam.force_registry.register(mock_force)
        assert len(beam.force_registry) == 2

        # Clear all forces
        beam.force_registry.clear()
        assert len(beam.force_registry) == 0


class TestInputFunctionComposition:
    """Test input function composition functionality."""

    def test_default_input_function(self, beam_file):
        """Test default input function behavior."""
        beam = DynamicEulerBernoulliBeam(beam_file)
        beam.create_input_func()

        assert beam.input_func is not None

        # Test input function evaluation
        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01
        test_input = np.random.rand(n_dofs) * 0.1

        result = beam.input_func(test_state, test_input)
        assert result.shape == (2 * n_dofs,)

    def test_external_input_processor(self, beam_file):
        """Test external input processing function."""
        beam = DynamicEulerBernoulliBeam(beam_file)

        def custom_input_processor(x, u, t):
            """Custom input processor that scales input by 2."""
            return u * 2.0  # Return modification to be added to original

        beam.create_input_func(custom_input_processor)

        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01
        test_input = np.random.rand(n_dofs) * 0.1

        result = beam.input_func(test_state, test_input)
        assert result.shape == (2 * n_dofs,)

        # Compare with default input processor
        beam_default = DynamicEulerBernoulliBeam(beam_file)
        beam_default.create_input_func()
        result_default = beam_default.input_func(test_state, test_input)

        # Should be different due to custom processing
        assert not np.allclose(result, result_default)

    def test_input_registry_functionality(self, beam_file):
        """Test input registry registration and aggregation."""
        beam = DynamicEulerBernoulliBeam(beam_file)

        # Register input handlers
        handler1 = MockInputHandler(beam, gain=0.1)
        handler2 = MockInputHandler(beam, gain=0.2)

        beam.input_registry.register(handler1)
        beam.input_registry.register(handler2)

        assert len(beam.input_registry) == 2

        # Create input function using registry
        beam.create_input_func()

        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01
        test_input = np.ones(n_dofs)  # Unit input for easy calculation

        result = beam.input_func(test_state, test_input)

        # The aggregated function should add both handler contributions
        # handler1: u * 0.1, handler2: u * 0.2
        # Total modification: u * 0.3
        # Final processed input: u + u * 0.3 = u * 1.3
        expected_processed_input = test_input * 1.3

        # Compare with manual calculation
        beam_manual = DynamicEulerBernoulliBeam(beam_file)
        beam_manual.create_input_func(lambda x, u, t: expected_processed_input)
        result_manual = beam_manual.input_func(test_state, test_input)

        assert np.allclose(result, result_manual)


class TestForceRegistryManagement:
    """Test force registry management functionality."""

    def test_force_registry_initialization(self, beam_file):
        """Test force registry initialization."""
        beam = DynamicEulerBernoulliBeam(beam_file)

        assert isinstance(beam.force_registry, ForceRegistry)
        assert isinstance(beam.input_registry, InputRegistry)
        assert len(beam.force_registry) == 0
        assert len(beam.input_registry) == 0

    def test_force_registry_contains(self, beam_file):
        """Test __contains__ method of force registry."""
        beam = DynamicEulerBernoulliBeam(beam_file)
        mock_force = MockForce(beam)

        assert mock_force not in beam.force_registry
        beam.force_registry.register(mock_force)
        assert mock_force in beam.force_registry

    def test_get_registered_forces(self, beam_file):
        """Test getting list of registered forces."""
        beam = DynamicEulerBernoulliBeam(beam_file)

        mock_force1 = MockForce(beam, force_magnitude=100.0)
        mock_force2 = MockForce(beam, force_magnitude=200.0)

        beam.force_registry.register(mock_force1)
        beam.force_registry.register(mock_force2)

        forces = beam.force_registry.get_registered_forces()
        assert len(forces) == 2
        assert mock_force1 in forces
        assert mock_force2 in forces

        # Should return a copy, not the original list
        forces.clear()
        assert len(beam.force_registry) == 2

    def test_disabled_force_not_registered(self, beam_file):
        """Test that disabled forces are not registered."""
        beam = DynamicEulerBernoulliBeam(beam_file)

        disabled_force = MockForce(beam, enabled=False)
        beam.force_registry.register(disabled_force)

        # Should not be registered because it's disabled
        assert len(beam.force_registry) == 0
        assert disabled_force not in beam.force_registry


class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""

    def test_system_func_creation_before_calling(self, beam_file):
        """Test error when trying to get system function before creation."""
        beam = DynamicEulerBernoulliBeam(beam_file)

        with pytest.raises(RuntimeError, match="System function not yet created"):
            beam.get_system_func()

    def test_dynamic_system_creation_incomplete(self, beam_file):
        """Test error when trying to get dynamic system with incomplete setup."""
        beam = DynamicEulerBernoulliBeam(beam_file)
        beam.create_system_func()  # Only create system func, not input func

        with pytest.raises(
            RuntimeError, match="System and input functions must be created first"
        ):
            beam.get_dynamic_system()

    def test_empty_force_registry_function(self, beam_file):
        """Test that empty force registry creates valid zero-force function."""
        beam = DynamicEulerBernoulliBeam(beam_file)

        # Create aggregated function from empty registry
        forces_func = beam.force_registry.create_aggregated_function()

        n_dofs = len(beam.state_to_node_param) // 2
        test_state = np.random.rand(2 * n_dofs) * 0.01

        result = forces_func(test_state, 0.0)
        expected = np.zeros(n_dofs)

        assert np.allclose(result, expected)

    def test_large_system_evaluation(self, beam_file):
        """Test system evaluation with larger state vectors."""
        beam = DynamicEulerBernoulliBeam(beam_file)
        beam.create_system_func()
        beam.create_input_func()

        n_dofs = len(beam.state_to_node_param) // 2
        large_state = np.random.rand(2 * n_dofs) * 0.1

        # Should handle larger state vectors without issues
        result = beam.get_system_func()(large_state)
        assert result.shape == (2 * n_dofs,)
        assert np.isfinite(result).all()

    def test_integration_with_solve_ivp(self, beam_file):
        """Test integration with scipy's solve_ivp."""
        beam = DynamicEulerBernoulliBeam(beam_file)
        beam.create_system_func()
        beam.create_input_func()

        dynamic_system = beam.get_dynamic_system()

        n_dofs = len(beam.state_to_node_param) // 2
        initial_state = np.random.rand(2 * n_dofs) * 0.001  # Small initial conditions
        zero_input = np.zeros(n_dofs)

        # Test short integration
        t_span = (0, 0.01)
        t_eval = np.linspace(0, 0.01, 10)

        sol = solve_ivp(
            lambda t, x: dynamic_system(t, x, zero_input),
            t_span,
            initial_state,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
        )

        assert sol.success
        assert sol.y.shape == (2 * n_dofs, len(t_eval))
