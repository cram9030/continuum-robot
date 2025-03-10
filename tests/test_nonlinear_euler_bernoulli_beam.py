import pytest
import numpy as np
import tempfile
import os
import pandas as pd

from continuum_robot.models.nonlinear_euler_bernoulli_beam import (
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
    assert 7 in constrained_dofs  # w₃ from pinned


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


def test_dof_mapping_initialization(valid_parameters):
    """Test initial DOF mapping creation."""
    beam = NonlinearEulerBernoulliBeam(valid_parameters)

    # Check mappings were created
    assert hasattr(beam, "dof_to_node_param")
    assert hasattr(beam, "node_param_to_dof")

    # Check mapping for a 4-segment beam (5 nodes, 15 DOFs)
    assert len(beam.dof_to_node_param) == 15
    assert len(beam.node_param_to_dof) == 15

    # Check first node mappings
    assert beam.dof_to_node_param[0] == ("u", 0)
    assert beam.dof_to_node_param[1] == ("w", 0)
    assert beam.dof_to_node_param[2] == ("phi", 0)

    # Check last node mappings
    assert beam.dof_to_node_param[12] == ("u", 4)
    assert beam.dof_to_node_param[13] == ("w", 4)
    assert beam.dof_to_node_param[14] == ("phi", 4)

    # Check reverse mappings
    assert beam.node_param_to_dof[("u", 0)] == 0
    assert beam.node_param_to_dof[("w", 4)] == 13
    assert beam.node_param_to_dof[("phi", 4)] == 14


def test_dof_mapping_boundary_conditions(initialized_beam):
    """Test DOF mapping updates with boundary conditions."""
    beam = initialized_beam

    # Store original mappings
    orig_dof_to_node_param = beam.dof_to_node_param.copy()

    # Apply fixed boundary condition at first node
    beam.apply_boundary_conditions({0: BoundaryConditionType.FIXED})

    print(orig_dof_to_node_param)
    print(beam.dof_to_node_param)
    # Check mappings were updated
    assert len(beam.dof_to_node_param) == 12  # Original 15 - 3 constrained DOFs

    # Check node 0 DOFs are no longer in the mapping values
    assert ("u", 0) not in beam.dof_to_node_param.values()
    assert ("w", 0) not in beam.dof_to_node_param.values()
    assert ("phi", 0) not in beam.dof_to_node_param.values()

    # Check that previously node 1 DOFs are now at indices 0, 1, and 2
    assert beam.dof_to_node_param[0] == orig_dof_to_node_param[3]  # u at node 1
    assert beam.dof_to_node_param[1] == orig_dof_to_node_param[4]  # w at node 1
    assert beam.dof_to_node_param[2] == orig_dof_to_node_param[5]  # phi at node 1

    # Test the accessor methods
    assert beam.get_dof_to_node_param(0) == ("u", 1)
    assert beam.get_dof_index(1, "u") == 0


def test_dof_mapping_clear_boundary_conditions(initialized_beam):
    """Test DOF mapping restoration after clearing boundary conditions."""
    beam = initialized_beam

    # Store original mappings
    orig_dof_to_node_param = beam.dof_to_node_param.copy()
    orig_node_param_to_dof = beam.node_param_to_dof.copy()

    # Apply boundary conditions
    beam.apply_boundary_conditions({0: BoundaryConditionType.FIXED})

    # Check mappings were updated
    assert len(beam.dof_to_node_param) == 12

    # Clear boundary conditions
    beam.clear_boundary_conditions()

    # Check mappings were restored
    assert len(beam.dof_to_node_param) == 15
    assert beam.dof_to_node_param == orig_dof_to_node_param
    assert beam.node_param_to_dof == orig_node_param_to_dof


def test_dof_mapping_multiple_boundary_conditions(initialized_beam):
    """Test DOF mapping with multiple boundary conditions."""
    beam = initialized_beam
    print(beam.dof_to_node_param)

    # Apply multiple boundary conditions
    conditions = {
        0: BoundaryConditionType.FIXED,  # Constrains DOFs 0,1,2
        2: BoundaryConditionType.PINNED,  # Constrains DOFs 6,8
    }
    beam.apply_boundary_conditions(conditions)

    # Check mappings were updated correctly
    assert len(beam.dof_to_node_param) == 10  # Original 15 - 5 constrained DOFs

    print(beam.dof_to_node_param)

    # After remapping, check that nodes 0 and 2's constrained DOFs are gone
    assert ("u", 0) not in beam.dof_to_node_param.values()
    assert ("w", 0) not in beam.dof_to_node_param.values()
    assert ("phi", 0) not in beam.dof_to_node_param.values()
    assert ("u", 2) not in beam.dof_to_node_param.values()
    assert ("w", 2) not in beam.dof_to_node_param.values()

    # Test accessor methods
    assert beam.get_dof_to_node_param(2) == ("phi", 1)

    # Node 3's u DOF should have been remapped from 9 to a lower index
    node3_u_dof = beam.get_dof_index(3, "u")
    assert node3_u_dof < 9  # It should have been moved up in the mapping


def test_dof_access_errors(valid_parameters):
    """Test error handling in DOF mapping accessors."""
    beam = NonlinearEulerBernoulliBeam(valid_parameters)

    # Test invalid DOF index
    with pytest.raises(KeyError):
        beam.get_dof_to_node_param(20)

    # Test invalid parameter
    with pytest.raises(KeyError):
        beam.get_dof_index(0, "invalid_param")

    # Test invalid node
    with pytest.raises(KeyError):
        beam.get_dof_index(10, "u")
