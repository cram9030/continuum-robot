import pytest
import numpy as np
import tempfile
import os
import pandas as pd

from models.nonlinear_euler_bernoulli_beam import (
    NonlinearEulerBernoulliBeam,
    BoundaryConditionType,
)


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


@pytest.fixture
def initialized_beam(valid_parameters):
    """Create initialized beam with stiffness function and mass matrix."""
    beam = NonlinearEulerBernoulliBeam(valid_parameters)
    beam.create_stiffness_function()
    beam.create_mass_matrix()
    return beam


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


def test_mass_matrix(nitinol_file):
    """Test mass matrix creation and properties."""
    beam = NonlinearEulerBernoulliBeam(nitinol_file)

    # Test matrix creation
    beam.create_mass_matrix()
    M = beam.get_mass_matrix()

    # Test matrix properties
    assert isinstance(M, np.ndarray)
    assert M.shape == (15, 15)  # 4 segments * 3 states + 3 states for end node
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


def test_boundary_condition_prerequisites(nitinol_file):
    """Test error handling when applying boundary conditions without prerequisites."""
    beam = NonlinearEulerBernoulliBeam(nitinol_file)

    # Try applying boundary conditions before creating stiffness function
    with pytest.raises(RuntimeError):
        beam.apply_boundary_conditions({0: BoundaryConditionType.FIXED})


def test_clear_boundary_conditions(nitinol_file):
    """Test clearing boundary conditions."""
    beam = NonlinearEulerBernoulliBeam(nitinol_file)
    beam.create_stiffness_function()

    # Store original stiffness function
    original_stiffness = beam.get_stiffness_function()
    state = np.ones(15)
    original_forces = original_stiffness(state)

    # Apply and then clear boundary conditions
    beam.apply_boundary_conditions({0: BoundaryConditionType.FIXED})
    beam.clear_boundary_conditions()

    # Check state is restored
    assert not beam.has_boundary_conditions()
    assert len(beam.get_boundary_conditions()) == 0
    assert len(beam.get_constrained_dofs()) == 0

    # Check forces match original
    new_forces = beam.get_stiffness_function()(state)
    assert np.allclose(original_forces, new_forces)


def test_stiffness_function_parallelization(initialized_beam):
    """Test parallel computation of stiffness function."""
    # Test pre-computed segment functions
    assert len(initialized_beam.segment_stiffness_functions) == 4
    assert all(callable(f) for f in initialized_beam.segment_stiffness_functions)

    # Test stiffness computation with different state vectors
    n_states = (len(initialized_beam.parameters) + 1) * 3

    # Test zero state
    x_zero = np.zeros(n_states)
    f_zero = initialized_beam.get_stiffness_function()(x_zero)
    assert isinstance(f_zero, np.ndarray)
    assert f_zero.shape == (n_states,)
    assert np.allclose(f_zero, np.zeros_like(f_zero))

    # Test unit state
    x_unit = np.ones(2 * n_states)
    f_unit = initialized_beam.get_stiffness_function()(x_unit)
    assert isinstance(f_unit, np.ndarray)
    assert f_unit.shape == (n_states,)
    assert not np.allclose(f_unit, np.zeros_like(f_unit))


def test_boundary_condition_dimension_reduction(initialized_beam):
    """Test boundary condition application with dimension reduction."""
    # Initial system size
    initial_size = (len(initialized_beam.parameters) + 1) * 3

    # Apply fixed boundary condition at first node
    conditions = {0: BoundaryConditionType.FIXED}
    initialized_beam.apply_boundary_conditions(conditions)

    # Test force calculation with reduced dimensions
    x_reduced = np.ones(initial_size - 3)  # 3 DOFs constrained
    f_reduced = initialized_beam.get_stiffness_function()(x_reduced)
    assert f_reduced.shape == (initial_size - 3,)

    # Test mass matrix reduction
    M_reduced = initialized_beam.get_mass_matrix()
    assert M_reduced.shape == (initial_size - 3, initial_size - 3)

    # Verify constrained DOFs
    assert initialized_beam.get_constrained_dofs() == {0, 1, 2}


def test_mixed_boundary_conditions(initialized_beam):
    """Test applying multiple types of boundary conditions."""
    conditions = {
        0: BoundaryConditionType.FIXED,  # Constrains all 3 DOFs
        2: BoundaryConditionType.PINNED,  # Constrains 2 DOFs
    }
    initialized_beam.apply_boundary_conditions(conditions)

    # Check constrained DOFs
    constrained_dofs = initialized_beam.get_constrained_dofs()
    assert len(constrained_dofs) == 5  # 3 from fixed + 2 from pinned

    # Verify specific DOFs constrained
    assert 0 in constrained_dofs  # u₁ from fixed
    assert 1 in constrained_dofs  # θ₁ from fixed
    assert 2 in constrained_dofs  # w₁ from fixed
    assert 6 in constrained_dofs  # u₃ from pinned
    assert 8 in constrained_dofs  # w₃ from pinned


def test_boundary_condition_clearing(initialized_beam):
    """Test clearing boundary conditions."""
    initial_size = (len(initialized_beam.parameters) + 1) * 3

    # Apply and then clear boundary conditions
    conditions = {0: BoundaryConditionType.FIXED}
    initialized_beam.apply_boundary_conditions(conditions)
    initialized_beam.clear_boundary_conditions()

    # Test restoration of original dimensions
    x_full = np.ones(initial_size)
    f_full = initialized_beam.get_stiffness_function()(x_full)
    assert f_full.shape == (initial_size,)

    # Verify boundary condition tracking cleared
    assert not initialized_beam.has_boundary_conditions()
    assert len(initialized_beam.get_boundary_conditions()) == 0
    assert len(initialized_beam.get_constrained_dofs()) == 0


def test_invalid_boundary_conditions(initialized_beam):
    """Test invalid boundary condition scenarios."""
    # Test invalid node index
    with pytest.raises(ValueError):
        initialized_beam.apply_boundary_conditions({-1: BoundaryConditionType.FIXED})

    # Test constraining all DOFs
    all_nodes = range(len(initialized_beam.parameters) + 1)
    all_fixed = {i: BoundaryConditionType.FIXED for i in all_nodes}
    with pytest.raises(ValueError):
        initialized_beam.apply_boundary_conditions(all_fixed)


def test_segment_force_combination(initialized_beam):
    """Test correct combination of segment forces into nodal forces."""
    n_segments = len(initialized_beam.parameters)
    n_states = (n_segments + 1) * 3

    # Create state vector with known pattern
    x = np.arange(n_states, dtype=float)

    # Get global forces
    f = initialized_beam.get_stiffness_function()(x)

    # First node should only have forces from first segment
    assert not np.allclose(f[0:3], 0)  # Should have forces

    # Internal nodes should have forces from two segments
    for i in range(1, n_segments):
        node_forces = f[3 * i : 3 * (i + 1)]
        assert not np.allclose(node_forces, 0)  # Should have combined forces

    # Last node should only have forces from last segment
    assert not np.allclose(f[-3:], 0)  # Should have forces
