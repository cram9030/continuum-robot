import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from scipy.integrate import solve_ivp

from dynamic_beam_model import DynamicEulerBernoulliBeam


@pytest.fixture
def beam_files():
    """Create temporary CSV files with linear and nonlinear beam parameters."""
    files = []

    # Create linear beam file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write(
            "length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition\n"
        )
        params = [
            (0.25, 75e9, 4.91e-10, 6450, 7.85e-5, "linear", bc)
            for bc in ["FIXED", "NONE", "NONE", "NONE"]
        ]
        for p in params:
            f.write(f"{','.join(str(x) for x in p)}\n")
        files.append(f.name)

    # Create nonlinear beam file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write(
            "length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition\n"
        )
        params = [
            (0.25, 75e9, 4.91e-10, 6450, 7.85e-5, "nonlinear", bc)
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
        lambda t, x: linear_system(t, x, u_linear), t_span, x0_linear
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
    print(nonlinear_beam.nonlinear_params)
    x0_nonlinear = np.zeros(2 * n_states)

    def u_nonlinear(t):
        return np.sin(t) * np.ones(n_states)

    sol_nonlinear = solve_ivp(
        lambda t, x: nonlinear_system(t, x, u_nonlinear), t_span, x0_nonlinear
    )
    assert sol_nonlinear.success
    assert sol_nonlinear.t[0] == t_span[0]
    assert sol_nonlinear.t[-1] == t_span[1]


def test_mixed_system_error():
    """Test error raised for mixed linear/nonlinear system."""
    # Create mixed system file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write(
            "length,elastic_modulus,moment_inertia,density,cross_area,type,boundary_condition\n"
        )
        params = [
            (0.25, 75e9, 4.91e-10, 6450, 7.85e-5, "linear", "FIXED"),
            (0.25, 75e9, 4.91e-10, 6450, 7.85e-5, "nonlinear", "NONE"),
        ]
        for p in params:
            f.write(f"{','.join(str(x) for x in p)}\n")

    beam = DynamicEulerBernoulliBeam(f.name)
    with pytest.raises(
        ValueError, match="Mixed linear/nonlinear systems not currently supported"
    ):
        beam.create_system_func()

    os.unlink(f.name)
