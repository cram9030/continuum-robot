import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from scipy.integrate import solve_ivp

from continuum_robot.models.dynamic_beam_model import (
    DynamicEulerBernoulliBeam,
    FluidDynamicsParams,
)


@pytest.fixture
def beam_files():
    """Create temporary CSV files with linear and nonlinear beam parameters."""
    files = []

    # Create linear beam file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write(
            "length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition,wetted_area,drag_coef\n"
        )
        params = [
            (0.25, 75e9, 4.91e-10, 6450, 7.85e-5, "linear", bc, 0.001, 0.5)
            for bc in ["FIXED", "NONE", "NONE", "NONE"]
        ]
        for p in params:
            f.write(f"{','.join(str(x) for x in p)}\n")
        files.append(f.name)

    # Create nonlinear beam file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write(
            "length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition,wetted_area,drag_coef\n"
        )
        params = [
            (0.25, 75e9, 4.91e-10, 6450, 7.85e-5, "nonlinear", bc, 0.001, 0.5)
            for bc in ["FIXED", "NONE", "NONE", "NONE"]
        ]
        for p in params:
            f.write(f"{','.join(str(x) for x in p)}\n")
        files.append(f.name)

    yield files

    # Cleanup temp files
    for f in files:
        os.unlink(f)


def test_initialization(beam_files):
    """Test initialization of linear and nonlinear beams."""
    linear_file, nonlinear_file = beam_files

    # Test linear beam
    linear_beam = DynamicEulerBernoulliBeam(linear_file)
    assert linear_beam is not None
    assert len(linear_beam.params) == 4
    assert not linear_beam.linear_params.empty
    assert linear_beam.nonlinear_params.empty

    # Test nonlinear beam
    nonlinear_beam = DynamicEulerBernoulliBeam(nonlinear_file)
    assert nonlinear_beam is not None
    assert len(nonlinear_beam.params) == 4
    assert nonlinear_beam.linear_params.empty
    assert not nonlinear_beam.nonlinear_params.empty


def test_fluid_params_initialization(beam_files):
    """Test initialization with fluid dynamics parameters."""
    linear_file = beam_files[0]

    # Test with default fluid params (disabled)
    beam_default = DynamicEulerBernoulliBeam(linear_file)
    assert not beam_default.fluid_params.enable_fluid_effects

    # Test with custom fluid params (enabled)
    fluid_params = FluidDynamicsParams(fluid_density=1000.0, enable_fluid_effects=True)
    beam_fluid = DynamicEulerBernoulliBeam(linear_file, fluid_params=fluid_params)
    assert beam_fluid.fluid_params.enable_fluid_effects
    assert beam_fluid.fluid_params.fluid_density == 1000.0


def test_invalid_fluid_params(beam_files):
    """Test initialization with invalid fluid parameters."""
    linear_file = beam_files[0]

    # Test with negative fluid density
    fluid_params = FluidDynamicsParams(fluid_density=-1.0, enable_fluid_effects=True)
    with pytest.raises(ValueError, match="Fluid density must be positive"):
        DynamicEulerBernoulliBeam(linear_file, fluid_params=fluid_params)


def test_invalid_file():
    """Test initialization with nonexistent file."""
    with pytest.raises(FileNotFoundError):
        DynamicEulerBernoulliBeam("nonexistent.csv")


def test_invalid_parameters(beam_files):
    """Test initialization with invalid parameters."""
    linear_file = beam_files[0]

    # Create file with invalid element type
    df = pd.read_csv(linear_file)
    df.loc[0, "type"] = "invalid"

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        df.to_csv(f.name, index=False)
        with pytest.raises(ValueError):
            DynamicEulerBernoulliBeam(f.name)
    os.unlink(f.name)


def test_missing_fluid_columns():
    """Test initialization with missing fluid-related columns."""
    # Create file without wetted_area and drag_coef columns
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv")
    try:
        temp_file.write(
            "length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition\n"
        )
        params = [(0.25, 75e9, 4.91e-10, 6450, 7.85e-5, "linear", "FIXED")]
        for p in params:
            temp_file.write(f"{','.join(str(x) for x in p)}\n")
        temp_file.close()  # Close the file to ensure it's written to disk

        with pytest.raises(ValueError, match="CSV must contain columns"):
            fluid_params = FluidDynamicsParams(
                fluid_density=-1.0, enable_fluid_effects=True
            )
            DynamicEulerBernoulliBeam(temp_file.name, fluid_params=fluid_params)
    finally:
        # Clean up
        os.unlink(temp_file.name)


def test_system_creation(beam_files):
    """Test system function creation for linear and nonlinear beams."""
    linear_file, nonlinear_file = beam_files

    # Test linear beam system
    linear_beam = DynamicEulerBernoulliBeam(linear_file)
    linear_beam.create_system_func()
    linear_beam.create_input_func()

    linear_system = linear_beam.get_dynamic_system()
    assert callable(linear_system)

    # Test nonlinear beam system
    nonlinear_beam = DynamicEulerBernoulliBeam(nonlinear_file)
    nonlinear_beam.create_system_func()
    nonlinear_beam.create_input_func()

    nonlinear_system = nonlinear_beam.get_dynamic_system()
    assert callable(nonlinear_system)

    # Test with constant force inputs
    n_states = len(linear_beam.linear_params) * 2
    x0_linear = np.zeros(2 * n_states)
    u_linear = np.ones(n_states)

    dx_linear = linear_system(0, x0_linear, u_linear)
    assert isinstance(dx_linear, np.ndarray)
    assert dx_linear.shape == x0_linear.shape

    n_states = len(nonlinear_beam.nonlinear_params) * 3
    x0_nonlinear = np.zeros(2 * n_states)
    u_nonlinear = np.ones(n_states)

    dx_nonlinear = nonlinear_system(0, x0_nonlinear, u_nonlinear)
    assert isinstance(dx_nonlinear, np.ndarray)
    assert dx_nonlinear.shape == x0_nonlinear.shape


def test_system_creation_with_fluid(beam_files):
    """Test system function creation with fluid dynamics enabled."""
    linear_file, nonlinear_file = beam_files

    # Create fluid params
    fluid_params = FluidDynamicsParams(fluid_density=1000.0, enable_fluid_effects=True)

    # Test linear beam with fluid dynamics
    linear_beam = DynamicEulerBernoulliBeam(linear_file, fluid_params=fluid_params)
    linear_beam.create_system_func()
    linear_beam.create_input_func()

    linear_system = linear_beam.get_dynamic_system()
    assert callable(linear_system)

    # Test nonlinear beam with fluid dynamics
    nonlinear_beam = DynamicEulerBernoulliBeam(
        nonlinear_file, fluid_params=fluid_params
    )
    nonlinear_beam.create_system_func()
    nonlinear_beam.create_input_func()

    nonlinear_system = nonlinear_beam.get_dynamic_system()
    assert callable(nonlinear_system)


def test_solve_ivp_integration(beam_files):
    """Test solve_ivp integration for linear and nonlinear beams."""
    linear_file, nonlinear_file = beam_files
    t_span = [0, 0.1]

    # Test linear system integration
    linear_beam = DynamicEulerBernoulliBeam(linear_file)
    linear_beam.create_system_func()
    linear_beam.create_input_func()
    linear_system = linear_beam.get_dynamic_system()

    n_states = len(linear_beam.linear_params) * 2
    x0_linear = np.zeros(2 * n_states)

    def u_linear(t):
        return np.sin(t) * np.ones(n_states)

    sol_linear = solve_ivp(
        lambda t, x: linear_system(t, x, u_linear(t)), t_span, x0_linear
    )
    assert sol_linear.success
    assert sol_linear.t[0] == t_span[0]
    assert sol_linear.t[-1] == t_span[1]

    # Test nonlinear system integration
    nonlinear_beam = DynamicEulerBernoulliBeam(nonlinear_file)
    nonlinear_beam.create_system_func()
    nonlinear_beam.create_input_func()
    nonlinear_system = nonlinear_beam.get_dynamic_system()

    n_states = len(nonlinear_beam.nonlinear_params) * 3
    x0_nonlinear = np.zeros(2 * n_states)

    def u_nonlinear(t):
        return np.sin(t) * np.ones(n_states)

    sol_nonlinear = solve_ivp(
        lambda t, x: nonlinear_system(t, x, u_nonlinear(t)), t_span, x0_nonlinear
    )
    assert sol_nonlinear.success
    assert sol_nonlinear.t[0] == t_span[0]
    assert sol_nonlinear.t[-1] == t_span[1]


def test_solve_linear_beam_ivp_with_fluid(beam_files):
    """Test solve_ivp integration with fluid dynamics effects."""
    linear_file = beam_files[0]
    t_span = [0, 0.1]

    # Create fluid params
    fluid_params = FluidDynamicsParams(fluid_density=1000.0, enable_fluid_effects=True)

    # Test linear system with fluid dynamics
    linear_beam = DynamicEulerBernoulliBeam(linear_file, fluid_params=fluid_params)
    linear_beam.create_system_func()
    linear_beam.create_input_func()
    linear_system = linear_beam.get_dynamic_system()

    n_states = len(linear_beam.linear_params) * 2
    x0_linear = np.zeros(2 * n_states)

    def u_linear(t):
        return np.sin(t) * np.ones(n_states)

    sol_linear = solve_ivp(
        lambda t, x: linear_system(t, x, u_linear(t)), t_span, x0_linear
    )
    assert sol_linear.success

    # Ensure fluid effects have impact: compare with no fluid case
    linear_beam_no_fluid = DynamicEulerBernoulliBeam(linear_file)
    linear_beam_no_fluid.create_system_func()
    linear_beam_no_fluid.create_input_func()
    linear_system_no_fluid = linear_beam_no_fluid.get_dynamic_system()

    sol_no_fluid = solve_ivp(
        lambda t, x: linear_system_no_fluid(t, x, u_linear(t)), t_span, x0_linear
    )

    # Solutions should be different due to fluid effects
    # We compare the last state values
    assert not np.allclose(sol_linear.y[:, -1], sol_no_fluid.y[:, -1])


def test_mixed_system_error():
    """Test error raised for mixed linear/nonlinear system."""
    # Create mixed system file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write(
            "length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition,wetted_area,drag_coef\n"
        )
        params = [
            (0.25, 75e9, 4.91e-10, 6450, 7.85e-5, "linear", "FIXED", 0.001, 0.5),
            (0.25, 75e9, 4.91e-10, 6450, 7.85e-5, "nonlinear", "NONE", 0.001, 0.5),
        ]
        for p in params:
            f.write(f"{','.join(str(x) for x in p)}\n")

    beam = DynamicEulerBernoulliBeam(f.name)
    with pytest.raises(
        ValueError, match="Mixed linear/nonlinear systems not currently supported"
    ):
        beam.create_system_func()

    os.unlink(f.name)


def test_solve_nonlinear_with_fluid(beam_files):
    """Test solve_ivp integration with fluid dynamics for nonlinear beam model."""
    nonlinear_file = beam_files[1]  # Use the nonlinear beam file
    t_span = [0, 0.1]

    # Create fluid params
    fluid_params = FluidDynamicsParams(fluid_density=1000.0, enable_fluid_effects=True)

    # Test nonlinear system with fluid dynamics
    nonlinear_beam_fluid = DynamicEulerBernoulliBeam(
        nonlinear_file, fluid_params=fluid_params
    )
    nonlinear_beam_fluid.create_system_func()
    nonlinear_beam_fluid.create_input_func()
    nonlinear_system_fluid = nonlinear_beam_fluid.get_dynamic_system()

    n_states = len(nonlinear_beam_fluid.nonlinear_params) * 3
    x0_nonlinear = np.zeros(2 * n_states)

    def u_nonlinear(t):
        return np.sin(t) * np.ones(n_states)

    # Solve with fluid dynamics enabled
    sol_fluid = solve_ivp(
        lambda t, x: nonlinear_system_fluid(t, x, u_nonlinear(t)), t_span, x0_nonlinear
    )
    assert sol_fluid.success
    assert sol_fluid.t[0] == t_span[0]
    assert sol_fluid.t[-1] == t_span[1]

    # Test the same nonlinear system without fluid dynamics for comparison
    nonlinear_beam_no_fluid = DynamicEulerBernoulliBeam(nonlinear_file)
    nonlinear_beam_no_fluid.create_system_func()
    nonlinear_beam_no_fluid.create_input_func()
    nonlinear_system_no_fluid = nonlinear_beam_no_fluid.get_dynamic_system()

    # Solve without fluid dynamics
    sol_no_fluid = solve_ivp(
        lambda t, x: nonlinear_system_no_fluid(t, x, u_nonlinear(t)),
        t_span,
        x0_nonlinear,
    )
    assert sol_no_fluid.success

    # Compare results - solutions should be different due to fluid effects
    # We compare both positions and velocities at the last time step
    assert not np.allclose(sol_fluid.y[:, -1], sol_no_fluid.y[:, -1], rtol=1e-5)

    # Check that fluid effects cause more damping (slower movement)
    # Extract velocities (second half of state vector)
    vel_fluid = np.linalg.norm(sol_fluid.y[n_states:, -1])
    vel_no_fluid = np.linalg.norm(sol_no_fluid.y[n_states:, -1])

    # Fluid dynamics should cause damping, resulting in lower velocities
    assert vel_fluid < vel_no_fluid, "Fluid dynamics should cause damping effect"

    # Test with different fluid densities to ensure correct scaling
    fluid_params_dense = FluidDynamicsParams(
        fluid_density=2000.0, enable_fluid_effects=True
    )
    nonlinear_beam_dense = DynamicEulerBernoulliBeam(
        nonlinear_file, fluid_params=fluid_params_dense
    )
    nonlinear_beam_dense.create_system_func()
    nonlinear_beam_dense.create_input_func()

    sol_dense = solve_ivp(
        lambda t, x: nonlinear_beam_dense.get_dynamic_system()(t, x, u_nonlinear(t)),
        t_span,
        x0_nonlinear,
    )

    # Higher fluid density should cause even more damping
    vel_dense = np.linalg.norm(sol_dense.y[n_states:, -1])
    assert vel_dense < vel_fluid, "Higher fluid density should cause more damping"


@pytest.fixture
def beam_file_with_fluid():
    """Create a temporary CSV file with beam parameters including fluid dynamics parameters."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write(
            "length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition,wetted_area,drag_coef\n"
        )
        params = [
            (0.25, 75e9, 4.91e-10, 6450, 7.85e-5, "linear", bc, 0.001, 1.2)
            for bc in ["FIXED", "NONE", "NONE", "NONE"]
        ]
        for p in params:
            f.write(f"{','.join(str(x) for x in p)}\n")
        linear_file = f.name

    yield linear_file
    os.unlink(linear_file)


def test_state_mapping_initialization(beam_files):
    """Test state mapping initialization for linear and nonlinear models."""
    linear_file, nonlinear_file = beam_files

    # Test linear beam state mapping
    linear_beam = DynamicEulerBernoulliBeam(linear_file)

    # Check mappings were created
    assert hasattr(linear_beam, "state_to_node_param")
    assert hasattr(linear_beam, "node_param_to_state")

    # For linear beam with 4 segments and fixed boundary condition at node 0:
    # - We have 5 nodes total
    # - Node 0 has boundary conditions (fixed), so it is removed
    # - 4 nodes remain, each with 2 DOFs (w, phi) = 8 position DOFs
    # - Total state size is 2 * 8 = 16 (8 positions + 8 velocities)
    assert len(linear_beam.state_to_node_param) == 16

    # Check mappings for positions and velocities
    for i in range(8):  # 8 position states
        param, node = linear_beam.state_to_node_param[i]
        assert param in ["w", "phi"]  # Either displacement or rotation

    for i in range(8, 16):  # 8 velocity states
        param, node = linear_beam.state_to_node_param[i]
        assert param in ["dw_dt", "dphi_dt"]  # Velocity of displacement or rotation

    # Test nonlinear beam state mapping
    nonlinear_beam = DynamicEulerBernoulliBeam(nonlinear_file)

    # For nonlinear beam with 4 segments and fixed boundary condition at node 0:
    # - We have 5 nodes total
    # - Node 0 has boundary conditions (fixed), so it is removed
    # - 4 nodes remain, each with 3 DOFs (u, w, phi) = 12 position DOFs
    # - Total state size is 2 * 12 = 24 (12 positions + 12 velocities)
    assert len(nonlinear_beam.state_to_node_param) == 24

    # Check mappings for positions and velocities
    for i in range(12):  # 12 position states
        param, node = nonlinear_beam.state_to_node_param[i]
        assert param in ["u", "w", "phi"]  # Axial, transverse, or rotation

    for i in range(12, 24):  # 12 velocity states
        param, node = nonlinear_beam.state_to_node_param[i]
        assert param in ["du_dt", "dw_dt", "dphi_dt"]  # Velocity components


def test_state_mapping_accessors(beam_files):
    """Test state mapping accessor methods."""
    linear_file, _ = beam_files
    beam = DynamicEulerBernoulliBeam(linear_file)

    # Get some example mappings
    idx_example = 2  # Some position index
    param, node = beam.get_state_to_node_param(idx_example)

    # Test reverse mapping
    assert beam.get_state_index(node, param) == idx_example

    # Test velocity parameter mapping
    vel_idx = idx_example + len(beam.state_to_node_param) // 2
    vel_param, vel_node = beam.get_state_to_node_param(vel_idx)

    assert vel_param.startswith("d") and vel_param.endswith("_dt")
    assert vel_param == f"d{param}_dt"
    assert vel_node == node

    # Test accessor with velocity parameter
    assert beam.get_state_index(vel_node, vel_param) == vel_idx

    # Test getter functions for the full mappings
    state_mapping = beam.get_state_mapping()
    node_param_mapping = beam.get_node_param_mapping()

    assert len(state_mapping) == len(beam.state_to_node_param)
    assert len(node_param_mapping) == len(beam.node_param_to_state)
    assert state_mapping[idx_example] == (param, node)
    assert node_param_mapping[(param, node)] == idx_example


def test_state_mapping_with_boundary_conditions(beam_files):
    """Test state mapping with boundary conditions."""
    linear_file, _ = beam_files
    beam = DynamicEulerBernoulliBeam(linear_file)

    # First node (0) already has fixed boundary condition, check 2nd and 3rd nodes
    node1_w_pos_idx = beam.get_state_index(1, "w")
    node1_w_vel_idx = beam.get_state_index(1, "dw_dt")

    # Check that w and dw_dt for node 1 are correctly mapped
    assert beam.get_state_to_node_param(node1_w_pos_idx) == ("w", 1)
    assert beam.get_state_to_node_param(node1_w_vel_idx) == ("dw_dt", 1)

    # The state indices should be offset by n_pos_states
    n_pos_states = len(beam.state_to_node_param) // 2
    assert node1_w_vel_idx == node1_w_pos_idx + n_pos_states


def test_fluid_coefficients_mapping(beam_file_with_fluid):
    """Test fluid coefficient mapping to transverse DOFs only."""
    fluid_params = FluidDynamicsParams(fluid_density=1000.0, enable_fluid_effects=True)
    beam = DynamicEulerBernoulliBeam(beam_file_with_fluid, fluid_params=fluid_params)

    # Check fluid coefficients were computed
    assert beam.fluid_coefficients is not None

    # Check that the drag factors were computed only for 'w' DOFs
    assert "w_vel_indices" in beam.fluid_coefficients
    assert "drag_factors" in beam.fluid_coefficients

    w_vel_indices = beam.fluid_coefficients["w_vel_indices"]
    w_pos_indices = beam.fluid_coefficients["w_pos_indices"]

    # Check that all velocity indices correspond to 'w' parameters
    for idx in w_vel_indices:
        param, node = beam.get_state_to_node_param(idx)
        assert param == "dw_dt"

    # Check that all position indices correspond to 'w' parameters
    for idx in w_pos_indices:
        param, node = beam.get_state_to_node_param(idx)
        assert param == "w"

    # The number of velocity indices should match the number of position indices
    assert len(w_vel_indices) == len(w_pos_indices)

    # Create system function and check if it uses the mapping
    beam.create_system_func()
    assert beam.system_func is not None

    # Run dynamic system to confirm it doesn't crash
    beam.create_input_func()
    dynamic_system = beam.get_dynamic_system()

    # Set up a test state vector
    n_states = len(beam.state_to_node_param)
    test_state = np.ones(n_states)

    # Apply a constant force
    test_force = np.zeros(n_states // 2)

    # This should run without error
    derivative = dynamic_system(0.0, test_state, test_force)

    # The derivative should have the right dimension
    assert len(derivative) == n_states


def test_state_mapping_errors(beam_files):
    """Test error handling in state mapping accessors."""
    linear_file, _ = beam_files
    beam = DynamicEulerBernoulliBeam(linear_file)

    # Test invalid state index
    with pytest.raises(KeyError):
        beam.get_state_to_node_param(100)

    # Test invalid parameter
    with pytest.raises(KeyError):
        beam.get_state_index(1, "invalid_param")

    # Test invalid node
    with pytest.raises(KeyError):
        beam.get_state_index(100, "w")
