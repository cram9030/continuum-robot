import pytest
import numpy as np
import tempfile
import os

from linear_euler_bernoulli_beam import LinearEulerBernoulliBeam, BoundaryConditionType


@pytest.fixture
def nitinol_file():
    """Create a temporary CSV file with Nitinol beam parameters."""
    # Create temp file with example Nitinol parameters
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        # Header
        f.write("length,elastic_modulus,moment_inertia,density,cross_area\n")

        # Example Nitinol parameters (SI units)
        # E = 75e9 Pa
        # ρ = 6450 kg/m³
        # r = 0.005 m
        # I = πr⁴/4 ≈ 4.91e-10 m⁴
        # A = πr² ≈ 7.85e-5 m²

        # Split 1m beam into 4 sections
        for _ in range(4):
            f.write("0.25,75e9,4.91e-10,6450,7.85e-5\n")

    yield f.name
    # Cleanup temp file
    os.unlink(f.name)


@pytest.fixture
def invalid_file():
    """Create a temporary CSV file with invalid parameters."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("length,elastic_modulus,moment_inertia,density,cross_area\n")
        f.write("0.25,75e9,-4.91e-10,6450,7.85e-5\n")  # Negative moment of inertia

    yield f.name
    os.unlink(f.name)


def test_initialization(nitinol_file: str):
    """Test basic initialization with valid file."""
    beam = LinearEulerBernoulliBeam(nitinol_file)
    assert beam is not None
    assert len(beam.parameters) == 4
    assert beam.get_length() == pytest.approx(1.0)


def test_invalid_file():
    """Test initialization with nonexistent file."""
    with pytest.raises(FileNotFoundError):
        LinearEulerBernoulliBeam("nonexistent.csv")


def test_invalid_parameters(invalid_file: str):
    """Test initialization with invalid parameters."""
    with pytest.raises(ValueError):
        LinearEulerBernoulliBeam(invalid_file)


def test_stiffness_matrix(nitinol_file: str):
    """Test stiffness matrix creation and retrieval."""
    beam = LinearEulerBernoulliBeam(nitinol_file)

    # Test matrix creation
    beam.create_stiffness_matrix()
    K = beam.get_stiffness_matrix()

    # Check matrix properties
    assert isinstance(K, np.ndarray)
    assert K.shape == (10, 10)  # 4 segments = 10x10 matrix
    assert np.allclose(K, K.T)  # Should be symmetric

    # Test single segment retrieval
    K_segment = beam.get_segment_stiffness(0)
    assert K_segment.shape == (4, 4)
    assert np.allclose(K_segment, K_segment.T)  # Should be symmetric


def test_mass_matrix(nitinol_file: str):
    """Test mass matrix creation and retrieval."""
    beam = LinearEulerBernoulliBeam(nitinol_file)

    # Test matrix creation
    beam.create_mass_matrix()
    M = beam.get_mass_matrix()

    # Check matrix properties
    assert isinstance(M, np.ndarray)
    assert M.shape == (10, 10)  # 4 segments = 10x10 matrix
    assert np.allclose(M, M.T)  # Should be symmetric

    # Test single segment retrieval
    M_segment = beam.get_segment_mass(0)
    assert M_segment.shape == (4, 4)
    assert np.allclose(M_segment, M_segment.T)  # Should be symmetric


def test_matrix_access_before_creation(nitinol_file: str):
    """Test accessing matrices before they're created."""
    beam = LinearEulerBernoulliBeam(nitinol_file)

    with pytest.raises(RuntimeError):
        beam.get_stiffness_matrix()

    with pytest.raises(RuntimeError):
        beam.get_mass_matrix()


def test_segment_index_bounds(nitinol_file: str):
    """Test segment access with invalid indices."""
    beam = LinearEulerBernoulliBeam(nitinol_file)

    with pytest.raises(IndexError):
        beam.get_segment_stiffness(-1)

    with pytest.raises(IndexError):
        beam.get_segment_stiffness(4)

    with pytest.raises(IndexError):
        beam.get_segment_mass(-1)

    with pytest.raises(IndexError):
        beam.get_segment_mass(4)


def test_parameter_update(nitinol_file: str):
    """Test matrix updates after reading new parameters."""
    beam = LinearEulerBernoulliBeam(nitinol_file)
    beam.create_stiffness_matrix()
    beam.create_mass_matrix()

    # Read same file again
    beam.read_parameter_file(nitinol_file)

    # Matrices should be reset
    with pytest.raises(RuntimeError):
        beam.get_stiffness_matrix()

    with pytest.raises(RuntimeError):
        beam.get_mass_matrix()


def test_matrix_values(nitinol_file: str):
    """Test specific matrix values for a single segment."""
    beam = LinearEulerBernoulliBeam(nitinol_file)
    beam.create_stiffness_matrix()
    beam.create_mass_matrix()

    # Get first segment matrices
    K = beam.get_stiffness_matrix()
    M = beam.get_mass_matrix()

    # Test key properties of matrices
    # K should be positive definite

    # M should be positive definite
    eigenvals = np.linalg.eigvals(M)
    assert np.all(eigenvals >= 0)

    # Remove the first two rows and first two columns of the stiffness matrix to emulate a cantilever beam boundary condition
    K_reduced = K[2:, 2:]
    eigenvals = np.linalg.eigvals(K_reduced)
    assert np.all(eigenvals >= 0)


@pytest.fixture
def beam_fixture(nitinol_file):
    """Create a beam with matrices initialized."""
    beam = LinearEulerBernoulliBeam(nitinol_file)
    beam.create_stiffness_matrix()
    beam.create_mass_matrix()
    return beam


def test_multiple_boundary_conditions(beam_fixture):
    """Test applying multiple boundary conditions simultaneously."""
    beam = beam_fixture

    # Store original matrix sizes
    orig_size = beam.get_stiffness_matrix().shape[0]

    # Apply multiple boundary conditions
    conditions = {
        0: BoundaryConditionType.FIXED,  # Constrains DOFs 0,1
        2: BoundaryConditionType.PINNED,  # Constrains DOF 4
    }
    beam.apply_boundary_conditions(conditions)

    # Check matrices were reduced
    K = beam.get_stiffness_matrix()
    M = beam.get_mass_matrix()
    expected_size = orig_size - 3  # Removed 3 DOFs
    assert K.shape == (expected_size, expected_size)
    assert M.shape == (expected_size, expected_size)

    # Check constrained DOFs
    constrained = beam.get_constrained_dofs()
    assert constrained == {0, 1, 4}


def test_clear_boundary_conditions(beam_fixture):
    """Test clearing boundary conditions."""
    beam = beam_fixture

    # Store original matrices
    K_orig = beam.get_stiffness_matrix().copy()
    M_orig = beam.get_mass_matrix().copy()

    # Apply and then clear boundary conditions
    conditions = {0: BoundaryConditionType.FIXED}
    beam.apply_boundary_conditions(conditions)
    beam.clear_boundary_conditions()

    # Check matrices are restored
    assert not beam.has_boundary_conditions()
    assert len(beam.get_boundary_conditions()) == 0
    assert len(beam.get_constrained_dofs()) == 0
    assert np.allclose(K_orig, beam.get_stiffness_matrix())
    assert np.allclose(M_orig, beam.get_mass_matrix())


def test_invalid_boundary_conditions(beam_fixture):
    """Test invalid boundary condition scenarios."""
    beam = beam_fixture

    # Test invalid node index
    with pytest.raises(ValueError):
        beam.apply_boundary_conditions({-1: BoundaryConditionType.FIXED})

    # Test constraining all DOFs
    n_nodes = len(beam.parameters) + 1
    all_fixed = {i: BoundaryConditionType.FIXED for i in range(n_nodes)}
    with pytest.raises(ValueError):
        beam.apply_boundary_conditions(all_fixed)


def test_matrix_reduction(beam_fixture):
    """Test proper reduction of matrices after boundary conditions."""
    beam = beam_fixture

    # Get original matrices
    K_orig = beam.get_stiffness_matrix()
    M_orig = beam.get_mass_matrix()

    # Apply boundary condition to middle node
    mid_node = len(beam.parameters) // 2
    beam.apply_boundary_conditions({mid_node: BoundaryConditionType.FIXED})

    # Check reduced matrices
    K = beam.get_stiffness_matrix()
    M = beam.get_mass_matrix()

    # Should have removed 2 DOFs
    assert K.shape[0] == K_orig.shape[0] - 2
    assert M.shape[0] == M_orig.shape[0] - 2

    # Check remaining matrix is properly formed
    assert np.all(K.diagonal() != 0)  # No zero diagonal entries
    assert np.all(M.diagonal() != 0)


def test_apply_before_matrix_creation(nitinol_file):
    """Test applying boundary conditions before creating matrices."""
    beam = LinearEulerBernoulliBeam(nitinol_file)

    conditions = {0: BoundaryConditionType.FIXED}
    with pytest.raises(RuntimeError):
        beam.apply_boundary_conditions(conditions)


def test_clear_before_matrix_creation(nitinol_file):
    """Test clearing boundary conditions before creating matrices."""
    beam = LinearEulerBernoulliBeam(nitinol_file)

    with pytest.raises(RuntimeError):
        beam.clear_boundary_conditions()
