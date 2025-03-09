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


def test_solve_ivp_with_fluid(beam_files):
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
