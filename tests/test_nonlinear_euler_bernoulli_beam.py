import pytest
import numpy as np
import tempfile
import os
import pandas as pd

from nonlinear_euler_bernoulli_beam import NonlinearEulerBernoulliBeam


@pytest.fixture
def nitinol_file():
    """Create a temporary CSV file with Nitinol beam parameters."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        # Header
        f.write("length,elastic_modulus,moment_inertia,density,cross_area\n")

        # Example Nitinol parameters (SI units)
        # E = 75e9 Pa (linear region)
        # ρ = 6450 kg/m³
        # r = 0.005 m
        # I = πr⁴/4 ≈ 4.91e-10 m⁴
        # A = πr² ≈ 7.85e-5 m²

        # Split 1m beam into 4 sections
        for _ in range(4):
            f.write("0.25,75e9,4.91e-10,6450,7.85e-5\n")

    yield f.name
    os.unlink(f.name)


@pytest.fixture
def invalid_file():
    """Create a temporary CSV file with invalid parameters."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("length,elastic_modulus,moment_inertia,density,cross_area\n")
        f.write("0.25,75e9,-4.91e-10,6450,7.85e-5\n")  # Negative moment of inertia

    yield f.name
    os.unlink(f.name)


@pytest.fixture
def valid_parameters():
    """Create a DataFrame with valid Nitinol beam parameters."""
    return pd.DataFrame(
        {
            "length": [0.25] * 4,
            "elastic_modulus": [75e9] * 4,
            "moment_inertia": [4.91e-10] * 4,
            "density": [6450] * 4,
            "cross_area": [7.85e-5] * 4,
        }
    )


def test_dataframe_initialization(valid_parameters):
    """Test initialization with valid DataFrame."""
    beam = NonlinearEulerBernoulliBeam(valid_parameters)
    assert beam is not None
    assert len(beam.parameters) == 4
    assert beam.get_length() == pytest.approx(1.0)


def test_invalid_dataframe():
    """Test initialization with invalid DataFrame."""
    # Missing required column
    invalid_df = pd.DataFrame(
        {
            "length": [0.25],
            "elastic_modulus": [75e9],
            "density": [6450],
            "cross_area": [7.85e-5],
        }
    )
    with pytest.raises(ValueError):
        NonlinearEulerBernoulliBeam(invalid_df)

    # Negative value
    invalid_df = pd.DataFrame(
        {
            "length": [0.25],
            "elastic_modulus": [75e9],
            "moment_inertia": [-4.91e-10],  # Negative
            "density": [6450],
            "cross_area": [7.85e-5],
        }
    )
    with pytest.raises(ValueError):
        NonlinearEulerBernoulliBeam(invalid_df)


def test_invalid_input_type():
    """Test initialization with invalid input type."""
    with pytest.raises(TypeError):
        NonlinearEulerBernoulliBeam([1, 2, 3])  # List instead of DataFrame or path


def test_dataframe_modification(valid_parameters):
    """Test that modifying input DataFrame doesn't affect beam."""
    beam = NonlinearEulerBernoulliBeam(valid_parameters)
    original_length = beam.get_length()

    # Modify input DataFrame
    valid_parameters.iloc[0, 0] = 0.5

    # Beam should be unchanged
    assert beam.get_length() == original_length


def test_parameter_update_with_dataframe(valid_parameters):
    """Test updating parameters with new DataFrame."""
    beam = NonlinearEulerBernoulliBeam(valid_parameters)
    beam.create_stiffness_function()
    beam.create_mass_matrix()

    # Update with new parameters
    new_parameters = valid_parameters.copy()
    new_parameters["length"] = [0.5] * 4

    beam.update_parameters(new_parameters)

    # Matrices should be reset
    with pytest.raises(RuntimeError):
        beam.get_stiffness_function()
    with pytest.raises(RuntimeError):
        beam.get_mass_matrix()


def test_initialization(nitinol_file):
    """Test basic initialization with valid file."""
    beam = NonlinearEulerBernoulliBeam(nitinol_file)
    assert beam is not None
    assert len(beam.parameters) == 4
    assert beam.get_length() == pytest.approx(1.0)


def test_invalid_file():
    """Test initialization with nonexistent file."""
    with pytest.raises(FileNotFoundError):
        NonlinearEulerBernoulliBeam("nonexistent.csv")


def test_invalid_parameters(invalid_file):
    """Test initialization with invalid parameters."""
    with pytest.raises(ValueError):
        NonlinearEulerBernoulliBeam(invalid_file)


def test_stiffness_function(nitinol_file):
    """Test stiffness function creation and evaluation."""
    beam = NonlinearEulerBernoulliBeam(nitinol_file)

    # Test function creation
    beam.create_stiffness_function()
    stiffness_func = beam.get_stiffness_function()
    assert callable(stiffness_func)

    # Test single segment stiffness function
    segment_func = beam.get_segment_stiffness(0)
    assert callable(segment_func)

    # Test function evaluation with zero state (should return zero force)
    state = np.zeros(6)  # [u₁, θ₁, w₁, u₂, θ₂, w₂]
    force = segment_func(state)
    assert isinstance(force, np.ndarray)
    assert force.shape == (6,)
    assert np.allclose(force, np.zeros(6))

    # Test function evaluation with unit state
    state = np.ones(6)
    force = segment_func(state)
    assert isinstance(force, np.ndarray)
    assert force.shape == (6,)
    assert not np.allclose(force, np.zeros(6))  # Forces should be non-zero
    assert np.all(np.isfinite(force))  # Forces should be finite


def test_mass_matrix(nitinol_file):
    """Test mass matrix creation and properties."""
    beam = NonlinearEulerBernoulliBeam(nitinol_file)

    # Test matrix creation
    beam.create_mass_matrix()
    M = beam.get_mass_matrix()

    # Test matrix properties
    assert isinstance(M, np.ndarray)
    assert M.shape == (30, 30)  # 4 segments * 6 states + 6 states for end node
    assert np.allclose(M, M.T)  # Should be symmetric

    # Test single segment mass matrix
    M_segment = beam.get_segment_mass(0)
    assert M_segment.shape == (6, 6)
    assert np.allclose(M_segment, M_segment.T)  # Should be symmetric

    # Test positive definiteness of segment mass matrix
    eigenvals = np.linalg.eigvals(M_segment)
    assert np.all(eigenvals >= 0)


def test_matrix_access_before_creation(nitinol_file):
    """Test accessing matrices before they're created."""
    beam = NonlinearEulerBernoulliBeam(nitinol_file)

    with pytest.raises(RuntimeError):
        beam.get_stiffness_function()

    with pytest.raises(RuntimeError):
        beam.get_mass_matrix()


def test_parameter_update(nitinol_file):
    """Test matrix updates after reading new parameters."""
    beam = NonlinearEulerBernoulliBeam(nitinol_file)
    beam.create_stiffness_function()
    beam.create_mass_matrix()

    # Read same file again
    beam.read_parameter_file(nitinol_file)

    # Matrices should be reset
    with pytest.raises(RuntimeError):
        beam.get_stiffness_function()

    with pytest.raises(RuntimeError):
        beam.get_mass_matrix()


def test_boundary_conditions(nitinol_file):
    """Test applying boundary conditions."""
    beam = NonlinearEulerBernoulliBeam(nitinol_file)
    beam.create_stiffness_function()
    beam.create_mass_matrix()

    # Create full state vector for 4 segments + end node
    # Format: [u1, θ1, w1, u2, θ2, w2, ..., un, θn, wn]
    state = np.ones(30)  # 5 nodes * 3 DOF per node = 30 total states

    # Get initial forces before any boundary conditions
    stiffness_func = beam.get_stiffness_function()
    initial_forces = stiffness_func(state)

    # Test pinned boundary condition at first node
    beam.apply_pinned_boundary(0)
    forces = beam.get_stiffness_function()(state)

    # Check that displacement forces are zero at first node
    assert forces[0] == 0  # u1 component should be zero
    assert forces[2] == 0  # w1 component should be zero
    # Check that rotation force is unchanged
    assert np.isclose(forces[1], initial_forces[1])  # θ1 should be unchanged
    # Check other forces are unchanged
    assert np.allclose(forces[3:], initial_forces[3:])

    # Reset beam for next test
    beam.create_stiffness_function()

    # Test clamped boundary condition at first node
    beam.apply_clamped_boundary(0)
    forces = beam.get_stiffness_function()(state)

    # Check all forces are zero at first node
    assert forces[0] == 0  # u1 component should be zero
    assert forces[1] == 0  # θ1 component should be zero
    assert forces[2] == 0  # w1 component should be zero
    # Check other forces are unchanged
    assert np.allclose(forces[3:], initial_forces[3:])

    # Test boundary condition at middle node
    beam.create_stiffness_function()  # Reset beam
    beam.apply_pinned_boundary(2)  # Apply to third node (index 2)
    forces = beam.get_stiffness_function()(state)

    # Check that displacement forces are zero at third node
    assert forces[6] == 0  # u3 component should be zero
    assert forces[8] == 0  # w3 component should be zero
    # Check that rotation force is unchanged
    assert np.isclose(forces[7], initial_forces[7])  # θ3 should be unchanged
    # Check other forces are unchanged
    assert np.allclose(forces[0:6], initial_forces[0:6])  # Previous nodes unchanged
    assert np.allclose(forces[9:], initial_forces[9:])  # Later nodes unchanged

    # Test boundary condition at end node
    beam.create_stiffness_function()  # Reset beam
    beam.apply_clamped_boundary(4)  # Apply to last node
    forces = beam.get_stiffness_function()(state)

    # Check all forces are zero at end node
    assert forces[27] == 0  # u5 component should be zero
    assert forces[28] == 0  # θ5 component should be zero
    assert forces[29] == 0  # w5 component should be zero
