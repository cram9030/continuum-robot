import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from continuum_robot.models.abstractions import BoundaryConditionType
from continuum_robot.models.euler_bernoulli_beam import EulerBernoulliBeam
from control_design.linear_quadratic_regulator import LinearQuadraticRegulator
from continuum_robot.control.full_state_linear import FullStateLinear


class TestEulerBernoulliBeamStiffnessMatrix:
    """Test the new get_stiffness_matrix method in EulerBernoulliBeam."""

    @pytest.fixture
    def linear_beam_csv(self):
        """Create temporary CSV file with linear beam parameters."""
        csv_content = """length,elastic_modulus,moment_inertia,density,cross_area,type
1.0,200e9,1e-6,7850,1e-4,linear
1.0,200e9,1e-6,7850,1e-4,linear
1.0,200e9,1e-6,7850,1e-4,linear
1.0,200e9,1e-6,7850,1e-4,linear"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def mixed_beam_csv(self):
        """Create temporary CSV file with mixed linear/nonlinear beam parameters."""
        csv_content = """length,elastic_modulus,moment_inertia,density,cross_area,type
1.0,200e9,1e-6,7850,1e-4,linear
1.0,200e9,1e-6,7850,1e-4,nonlinear
1.0,200e9,1e-6,7850,1e-4,linear
1.0,200e9,1e-6,7850,1e-4,linear"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def linear_beam(self, linear_beam_csv):
        """Create a linear beam for testing."""
        df = pd.read_csv(linear_beam_csv)
        beam = EulerBernoulliBeam(df)
        return beam

    @pytest.fixture
    def mixed_beam(self, mixed_beam_csv):
        """Create a mixed beam for testing."""
        df = pd.read_csv(mixed_beam_csv)
        beam = EulerBernoulliBeam(df)
        return beam

    def test_get_stiffness_matrix_linear_beam_success(self, linear_beam):
        """Test successful extraction of stiffness matrix from linear beam."""
        K = linear_beam.get_stiffness_matrix()

        n_dofs = 15  # 5 nodes × 3 DOFs each
        assert K.shape == (n_dofs, n_dofs)
        assert np.allclose(K, K.T)  # Should be symmetric

    def test_get_stiffness_matrix_mixed_beam_failure(self, mixed_beam):
        """Test that mixed beam raises error when extracting stiffness matrix."""
        with pytest.raises(
            ValueError,
            match="Cannot extract stiffness matrix from beam with nonlinear segments",
        ):
            mixed_beam.get_stiffness_matrix()

    def test_get_stiffness_matrix_with_boundary_conditions(self, linear_beam):
        """Test stiffness matrix extraction with boundary conditions applied."""
        # Apply fixed boundary condition at first node
        boundary_conditions = {0: BoundaryConditionType.FIXED}
        linear_beam.apply_boundary_conditions(boundary_conditions)

        K = linear_beam.get_stiffness_matrix()

        # Should have reduced dimensions after boundary conditions
        n_constrained = 3  # u, w, φ at node 0
        n_free = 15 - n_constrained
        assert K.shape == (n_free, n_free)
        assert np.allclose(K, K.T)  # Should still be symmetric

    def test_get_stiffness_matrix_before_mass_assembly_error(self, linear_beam_csv):
        """Test that error is raised if mass matrix hasn't been assembled."""
        # Create beam without triggering mass matrix assembly
        beam = EulerBernoulliBeam.__new__(
            EulerBernoulliBeam
        )  # Create without calling __init__
        beam.segments = []
        beam.M = None  # Simulate uninitialized state

        with pytest.raises(
            RuntimeError,
            match="Mass matrix must be assembled before extracting stiffness matrix",
        ):
            beam.get_stiffness_matrix()


class TestLinearQuadraticRegulator:
    """Test LinearQuadraticRegulator class with matrix inputs."""

    @pytest.fixture
    def linear_beam_csv(self):
        """Create temporary CSV file with linear beam parameters."""
        csv_content = """length,elastic_modulus,moment_inertia,density,cross_area,type
1.0,200e9,1e-6,7850,1e-4,linear
1.0,200e9,1e-6,7850,1e-4,linear
1.0,200e9,1e-6,7850,1e-4,linear
1.0,200e9,1e-6,7850,1e-4,linear"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def linear_beam(self, linear_beam_csv):
        """Create a linear beam for testing."""
        df = pd.read_csv(linear_beam_csv)
        beam = EulerBernoulliBeam(df)
        return beam

    def test_initialization_success(self, linear_beam):
        """Test successful initialization with valid matrices."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = linear_beam.get_mass_matrix()

        n_dofs = K_beam.shape[0]
        Q = np.eye(2 * n_dofs)  # State weighting (positions + velocities)
        R = np.eye(n_dofs)  # Control weighting

        lqr = LinearQuadraticRegulator(K_beam, M_beam, Q, R)

        assert np.array_equal(lqr.K_beam, K_beam)
        assert np.array_equal(lqr.M_beam, M_beam)
        assert np.array_equal(lqr.Q, Q)
        assert np.array_equal(lqr.R, R)

    def test_invalid_stiffness_matrix_not_square(self, linear_beam):
        """Test initialization failure with non-square stiffness matrix."""
        M_beam = linear_beam.get_mass_matrix()
        K_beam = np.ones((10, 15))  # Not square

        n_dofs = M_beam.shape[0]
        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        with pytest.raises(ValueError, match="Stiffness matrix must be square"):
            LinearQuadraticRegulator(K_beam, M_beam, Q, R)

    def test_invalid_mass_matrix_not_square(self, linear_beam):
        """Test initialization failure with non-square mass matrix."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = np.ones((10, 15))  # Not square

        n_dofs = K_beam.shape[0]
        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        with pytest.raises(ValueError, match="Mass matrix must be square"):
            LinearQuadraticRegulator(K_beam, M_beam, Q, R)

    def test_mismatched_matrix_dimensions(self, linear_beam):
        """Test initialization failure with mismatched matrix dimensions."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = np.eye(10)  # Wrong size

        n_dofs = K_beam.shape[0]
        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        with pytest.raises(
            ValueError,
            match="Stiffness and mass matrices must have the same dimensions",
        ):
            LinearQuadraticRegulator(K_beam, M_beam, Q, R)

    def test_invalid_q_matrix_not_square(self, linear_beam):
        """Test initialization failure with non-square Q matrix."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = linear_beam.get_mass_matrix()

        n_dofs = K_beam.shape[0]
        Q = np.ones((10, 15))  # Not square
        R = np.eye(n_dofs)

        with pytest.raises(ValueError, match="Q matrix must be square"):
            LinearQuadraticRegulator(K_beam, M_beam, Q, R)

    def test_invalid_r_matrix_not_square(self, linear_beam):
        """Test initialization failure with non-square R matrix."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = linear_beam.get_mass_matrix()

        n_dofs = K_beam.shape[0]
        Q = np.eye(2 * n_dofs)
        R = np.ones((10, 15))  # Not square

        with pytest.raises(ValueError, match="R matrix must be square"):
            LinearQuadraticRegulator(K_beam, M_beam, Q, R)

    def test_invalid_q_matrix_not_positive_semidefinite(self, linear_beam):
        """Test initialization failure with non-positive semidefinite Q matrix."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = linear_beam.get_mass_matrix()

        n_dofs = K_beam.shape[0]
        Q = -np.eye(2 * n_dofs)  # Negative definite
        R = np.eye(n_dofs)

        with pytest.raises(ValueError, match="Q matrix must be positive semidefinite"):
            LinearQuadraticRegulator(K_beam, M_beam, Q, R)

    def test_invalid_r_matrix_not_positive_definite(self, linear_beam):
        """Test initialization failure with non-positive definite R matrix."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = linear_beam.get_mass_matrix()

        n_dofs = K_beam.shape[0]
        Q = np.eye(2 * n_dofs)
        R = np.zeros((n_dofs, n_dofs))  # Not positive definite

        with pytest.raises(ValueError, match="R matrix must be positive definite"):
            LinearQuadraticRegulator(K_beam, M_beam, Q, R)

    def test_get_a_matrix_dimensions(self, linear_beam):
        """Test A matrix computation and dimensions."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = linear_beam.get_mass_matrix()

        n_dofs = K_beam.shape[0]
        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        lqr = LinearQuadraticRegulator(K_beam, M_beam, Q, R)
        A = lqr.get_A()

        # A should be 2n × 2n for state space representation
        assert A.shape == (2 * n_dofs, 2 * n_dofs)

        # Upper right block should be identity
        assert np.allclose(A[:n_dofs, n_dofs:], np.eye(n_dofs))

        # Upper left block should be zero
        assert np.allclose(A[:n_dofs, :n_dofs], np.zeros((n_dofs, n_dofs)))

    def test_get_b_matrix_dimensions(self, linear_beam):
        """Test B matrix computation and dimensions."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = linear_beam.get_mass_matrix()

        n_dofs = K_beam.shape[0]
        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        lqr = LinearQuadraticRegulator(K_beam, M_beam, Q, R)
        B = lqr.get_B()

        # B should be 2n × n for full actuation
        assert B.shape == (2 * n_dofs, n_dofs)

        # Upper block should be zero
        assert np.allclose(B[:n_dofs, :], np.zeros((n_dofs, n_dofs)))

    def test_compute_gain_matrix_dimensions(self, linear_beam):
        """Test gain matrix computation and dimensions."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = linear_beam.get_mass_matrix()

        n_dofs = K_beam.shape[0]
        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        lqr = LinearQuadraticRegulator(K_beam, M_beam, Q, R)
        K = lqr.compute_gain_matrix()

        # K should be n × 2n
        assert K.shape == (n_dofs, 2 * n_dofs)

    def test_compute_gain_matrix_stability(self, linear_beam):
        """Test that computed gain matrix results in stable closed-loop system."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = linear_beam.get_mass_matrix()

        n_dofs = K_beam.shape[0]
        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        lqr = LinearQuadraticRegulator(K_beam, M_beam, Q, R)

        A = lqr.get_A()
        B = lqr.get_B()
        K = lqr.compute_gain_matrix()

        # Check closed-loop stability
        A_cl = A - B @ K
        eigenvals = np.linalg.eigvals(A_cl)

        # All eigenvalues should have negative real parts
        assert np.all(np.real(eigenvals) < 0)

    def test_get_k_calls_compute_gain_matrix(self, linear_beam):
        """Test that get_K calls compute_gain_matrix if needed."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = linear_beam.get_mass_matrix()

        n_dofs = K_beam.shape[0]
        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        lqr = LinearQuadraticRegulator(K_beam, M_beam, Q, R)

        # Should compute gain matrix on first call
        K1 = lqr.get_K()

        # Should return cached result on second call
        K2 = lqr.get_K()

        assert K1 is K2

    def test_dimension_mismatch_q_matrix(self, linear_beam):
        """Test error when Q matrix dimension doesn't match state dimension."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = linear_beam.get_mass_matrix()

        n_dofs = K_beam.shape[0]
        Q = np.eye(20)  # Wrong dimension
        R = np.eye(n_dofs)

        lqr = LinearQuadraticRegulator(K_beam, M_beam, Q, R)

        with pytest.raises(
            ValueError, match="Q matrix dimension.*must match state dimension"
        ):
            lqr.compute_gain_matrix()

    def test_dimension_mismatch_r_matrix(self, linear_beam):
        """Test error when R matrix dimension doesn't match input dimension."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = linear_beam.get_mass_matrix()

        n_dofs = K_beam.shape[0]
        Q = np.eye(2 * n_dofs)
        R = np.eye(20)  # Wrong dimension

        lqr = LinearQuadraticRegulator(K_beam, M_beam, Q, R)

        with pytest.raises(
            ValueError, match="R matrix dimension.*must match input dimension"
        ):
            lqr.compute_gain_matrix()


class TestFullStateLinearIntegration:
    """Test integration between LinearQuadraticRegulator and FullStateLinear."""

    @pytest.fixture
    def linear_beam_csv(self):
        """Create temporary CSV file with linear beam parameters."""
        csv_content = """length,elastic_modulus,moment_inertia,density,cross_area,type
1.0,200e9,1e-6,7850,1e-4,linear
1.0,200e9,1e-6,7850,1e-4,linear"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def simple_linear_beam(self, linear_beam_csv):
        """Create a simple 2-segment linear beam for testing."""
        df = pd.read_csv(linear_beam_csv)
        beam = EulerBernoulliBeam(df)
        return beam

    def test_lqr_with_full_state_linear_controller(self, simple_linear_beam):
        """Test integration of LQR with FullStateLinear controller."""
        K_beam = simple_linear_beam.get_stiffness_matrix()
        M_beam = simple_linear_beam.get_mass_matrix()

        n_dofs = K_beam.shape[0]
        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        # Create LQR controller
        lqr = LinearQuadraticRegulator(K_beam, M_beam, Q, R)
        gain_matrix = lqr.compute_gain_matrix()

        # Create FullStateLinear controller with LQR gain
        controller = FullStateLinear(gain_matrix)

        # Test dimensions match
        assert controller.gain_matrix.shape == (n_dofs, 2 * n_dofs)

        # Test controller computation
        x = np.random.randn(2 * n_dofs)  # Random state
        r = np.random.randn(2 * n_dofs)  # Random reference

        u = controller.compute_input(x, r, 0.0)

        assert u.shape == (n_dofs,)

    def test_lqr_gain_matrix_properties(self, simple_linear_beam):
        """Test that LQR gain matrix has expected properties."""
        K_beam = simple_linear_beam.get_stiffness_matrix()
        M_beam = simple_linear_beam.get_mass_matrix()

        n_dofs = K_beam.shape[0]
        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        lqr = LinearQuadraticRegulator(K_beam, M_beam, Q, R)
        K = lqr.compute_gain_matrix()

        # Check dimensions
        assert K.shape == (n_dofs, 2 * n_dofs)

        # Check that gain is not all zeros
        assert not np.allclose(K, 0)

    def test_system_stability_with_boundary_conditions(self, simple_linear_beam):
        """Test system stability when boundary conditions are applied."""
        # Apply fixed boundary condition at first node
        boundary_conditions = {0: BoundaryConditionType.FIXED}
        simple_linear_beam.apply_boundary_conditions(boundary_conditions)

        # Get constrained matrices
        K_beam = simple_linear_beam.get_stiffness_matrix()
        M_beam = simple_linear_beam.get_mass_matrix()

        # Verify reduced dimensions
        constrained_dofs = simple_linear_beam.get_constrained_dofs()
        assert len(constrained_dofs) == 3  # u, w, φ at node 0

        total_dofs = 9  # 3 nodes × 3 DOFs originally
        free_dofs = total_dofs - len(constrained_dofs)
        assert K_beam.shape == (free_dofs, free_dofs)
        assert M_beam.shape == (free_dofs, free_dofs)

        # Test LQR with reduced system
        Q = np.eye(2 * free_dofs)
        R = np.eye(free_dofs)

        lqr = LinearQuadraticRegulator(K_beam, M_beam, Q, R)
        K = lqr.compute_gain_matrix()

        # Check reduced dimensions
        assert K.shape == (free_dofs, 2 * free_dofs)

        # Verify stability
        A = lqr.get_A()
        B = lqr.get_B()
        A_cl = A - B @ K
        eigenvals = np.linalg.eigvals(A_cl)
        assert np.all(np.real(eigenvals) < 0)
