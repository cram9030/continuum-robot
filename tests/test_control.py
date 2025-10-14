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

    def _create_state_space_matrices(self, linear_beam):
        """Helper function to create A and B matrices from beam."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = linear_beam.get_mass_matrix()
        n_dofs = K_beam.shape[0]

        # Construct A matrix
        A = np.zeros((2 * n_dofs, 2 * n_dofs))
        A[:n_dofs, n_dofs:] = np.eye(n_dofs)
        M_inv = np.linalg.inv(M_beam)
        A[n_dofs:, :n_dofs] = -M_inv @ K_beam

        # Construct B matrix (full actuation)
        B = np.zeros((2 * n_dofs, n_dofs))
        B[n_dofs:, :] = M_inv

        return A, B, n_dofs

    def test_initialization_success(self, linear_beam):
        """Test successful initialization with valid matrices."""
        A, B, n_dofs = self._create_state_space_matrices(linear_beam)

        Q = np.eye(2 * n_dofs)  # State weighting (positions + velocities)
        R = np.eye(n_dofs)  # Control weighting

        lqr = LinearQuadraticRegulator(A, B, Q, R)

        assert np.array_equal(lqr.A, A)
        assert np.array_equal(lqr.B, B)
        assert np.array_equal(lqr.Q, Q)
        assert np.array_equal(lqr.R, R)

    def test_invalid_a_matrix_not_square(self, linear_beam):
        """Test initialization failure with non-square A matrix."""
        A = np.ones((10, 15))  # Not square
        B = np.ones((10, 5))

        Q = np.eye(10)
        R = np.eye(5)

        with pytest.raises(ValueError, match="A matrix must be square"):
            LinearQuadraticRegulator(A, B, Q, R)

    def test_invalid_b_matrix_not_2d(self, linear_beam):
        """Test initialization failure with non-2D B matrix."""
        A = np.eye(10)
        B = np.ones(10)  # 1D instead of 2D

        Q = np.eye(10)
        R = np.eye(5)

        with pytest.raises(ValueError, match="B matrix must be 2-dimensional"):
            LinearQuadraticRegulator(A, B, Q, R)

    def test_mismatched_matrix_dimensions(self, linear_beam):
        """Test initialization failure with mismatched matrix dimensions."""
        A = np.eye(10)
        B = np.ones((15, 5))  # Wrong row dimension

        Q = np.eye(10)
        R = np.eye(5)

        with pytest.raises(
            ValueError,
            match="A and B matrices must have compatible dimensions",
        ):
            LinearQuadraticRegulator(A, B, Q, R)

    def test_invalid_q_matrix_not_square(self, linear_beam):
        """Test initialization failure with non-square Q matrix."""
        A, B, n_dofs = self._create_state_space_matrices(linear_beam)

        Q = np.ones((10, 15))  # Not square
        R = np.eye(n_dofs)

        with pytest.raises(ValueError, match="Q matrix must be square"):
            LinearQuadraticRegulator(A, B, Q, R)

    def test_invalid_r_matrix_not_square(self, linear_beam):
        """Test initialization failure with non-square R matrix."""
        A, B, n_dofs = self._create_state_space_matrices(linear_beam)

        Q = np.eye(2 * n_dofs)
        R = np.ones((10, 15))  # Not square

        with pytest.raises(ValueError, match="R matrix must be square"):
            LinearQuadraticRegulator(A, B, Q, R)

    def test_invalid_q_matrix_not_positive_semidefinite(self, linear_beam):
        """Test initialization failure with non-positive semidefinite Q matrix."""
        A, B, n_dofs = self._create_state_space_matrices(linear_beam)

        Q = -np.eye(2 * n_dofs)  # Negative definite
        R = np.eye(n_dofs)

        with pytest.raises(ValueError, match="Q matrix must be positive semidefinite"):
            LinearQuadraticRegulator(A, B, Q, R)

    def test_invalid_r_matrix_not_positive_definite(self, linear_beam):
        """Test initialization failure with non-positive definite R matrix."""
        A, B, n_dofs = self._create_state_space_matrices(linear_beam)

        Q = np.eye(2 * n_dofs)
        R = np.zeros((n_dofs, n_dofs))  # Not positive definite

        with pytest.raises(ValueError, match="R matrix must be positive definite"):
            LinearQuadraticRegulator(A, B, Q, R)

    def test_get_a_matrix_dimensions(self, linear_beam):
        """Test A matrix getter returns correct matrix."""
        A, B, n_dofs = self._create_state_space_matrices(linear_beam)

        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        lqr = LinearQuadraticRegulator(A, B, Q, R)
        A_returned = lqr.get_A()

        # Should return the same A matrix
        assert np.array_equal(A_returned, A)
        assert A_returned.shape == (2 * n_dofs, 2 * n_dofs)

    def test_get_b_matrix_dimensions(self, linear_beam):
        """Test B matrix getter returns correct matrix."""
        A, B, n_dofs = self._create_state_space_matrices(linear_beam)

        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        lqr = LinearQuadraticRegulator(A, B, Q, R)
        B_returned = lqr.get_B()

        # Should return the same B matrix
        assert np.array_equal(B_returned, B)
        assert B_returned.shape == (2 * n_dofs, n_dofs)

    def test_compute_gain_matrix_dimensions(self, linear_beam):
        """Test gain matrix computation and dimensions."""
        A, B, n_dofs = self._create_state_space_matrices(linear_beam)

        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        lqr = LinearQuadraticRegulator(A, B, Q, R)
        K, S = lqr.compute_gain_matrix()

        # K should be n × 2n
        assert K.shape == (n_dofs, 2 * n_dofs)
        # S should be 2n × 2n
        assert S.shape == (2 * n_dofs, 2 * n_dofs)

    def test_compute_gain_matrix_stability(self, linear_beam):
        """Test that computed gain matrix results in stable closed-loop system."""
        A, B, n_dofs = self._create_state_space_matrices(linear_beam)

        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        lqr = LinearQuadraticRegulator(A, B, Q, R)
        K, _ = lqr.compute_gain_matrix()

        # Check closed-loop stability
        A_cl = A - B @ K
        eigenvals = np.linalg.eigvals(A_cl)

        # All eigenvalues should have negative real parts
        assert np.all(np.real(eigenvals) < 0)

    def test_get_k_calls_compute_gain_matrix(self, linear_beam):
        """Test that get_K calls compute_gain_matrix if needed."""
        A, B, n_dofs = self._create_state_space_matrices(linear_beam)

        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        lqr = LinearQuadraticRegulator(A, B, Q, R)

        # Should compute gain matrix on first call
        K1 = lqr.get_K()

        # Should return cached result on second call
        K2 = lqr.get_K()

        assert K1 is K2

    def test_dimension_mismatch_q_matrix(self, linear_beam):
        """Test error when Q matrix dimension doesn't match state dimension."""
        A, B, n_dofs = self._create_state_space_matrices(linear_beam)

        Q = np.eye(20)  # Wrong dimension
        R = np.eye(n_dofs)

        lqr = LinearQuadraticRegulator(A, B, Q, R)

        with pytest.raises(
            ValueError, match="Q matrix dimension.*must match state dimension"
        ):
            lqr.compute_gain_matrix()

    def test_dimension_mismatch_r_matrix(self, linear_beam):
        """Test error when R matrix dimension doesn't match input dimension."""
        A, B, n_dofs = self._create_state_space_matrices(linear_beam)

        Q = np.eye(2 * n_dofs)
        R = np.eye(20)  # Wrong dimension

        lqr = LinearQuadraticRegulator(A, B, Q, R)

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

    def _create_state_space_matrices(self, linear_beam):
        """Helper function to create A and B matrices from beam."""
        K_beam = linear_beam.get_stiffness_matrix()
        M_beam = linear_beam.get_mass_matrix()
        n_dofs = K_beam.shape[0]

        # Construct A matrix
        A = np.zeros((2 * n_dofs, 2 * n_dofs))
        A[:n_dofs, n_dofs:] = np.eye(n_dofs)
        M_inv = np.linalg.inv(M_beam)
        A[n_dofs:, :n_dofs] = -M_inv @ K_beam

        # Construct B matrix (full actuation)
        B = np.zeros((2 * n_dofs, n_dofs))
        B[n_dofs:, :] = M_inv

        return A, B, n_dofs

    def test_lqr_with_full_state_linear_controller(self, simple_linear_beam):
        """Test integration of LQR with FullStateLinear controller."""
        A, B, n_dofs = self._create_state_space_matrices(simple_linear_beam)

        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        # Create LQR controller
        lqr = LinearQuadraticRegulator(A, B, Q, R)
        gain_matrix, _ = lqr.compute_gain_matrix()

        # Create FullStateLinear controller with LQR gain and B matrix
        controller = FullStateLinear(gain_matrix, B)

        # Test dimensions match
        assert controller.gain_matrix.shape == (n_dofs, 2 * n_dofs)
        assert controller.B.shape == (2 * n_dofs, n_dofs)

        # Test controller computation
        x = np.random.randn(2 * n_dofs)  # Random state
        r = np.random.randn(2 * n_dofs)  # Random reference

        u = controller.compute_input(x, r, 0.0)

        # Output should be force vector with dimension n_dofs (acceleration/force space)
        assert u.shape == (n_dofs,)

    def test_lqr_gain_matrix_properties(self, simple_linear_beam):
        """Test that LQR gain matrix has expected properties."""
        A, B, n_dofs = self._create_state_space_matrices(simple_linear_beam)

        Q = np.eye(2 * n_dofs)
        R = np.eye(n_dofs)

        lqr = LinearQuadraticRegulator(A, B, Q, R)
        K, _ = lqr.compute_gain_matrix()

        # Check dimensions
        assert K.shape == (n_dofs, 2 * n_dofs)

        # Check that gain is not all zeros
        assert not np.allclose(K, 0)

    def test_system_stability_with_boundary_conditions(self, simple_linear_beam):
        """Test system stability when boundary conditions are applied."""
        # Apply fixed boundary condition at first node
        boundary_conditions = {0: BoundaryConditionType.FIXED}
        simple_linear_beam.apply_boundary_conditions(boundary_conditions)

        # Get constrained matrices and create state-space representation
        A, B, free_dofs = self._create_state_space_matrices(simple_linear_beam)

        # Verify reduced dimensions
        constrained_dofs = simple_linear_beam.get_constrained_dofs()
        assert len(constrained_dofs) == 3  # u, w, φ at node 0

        total_dofs = 9  # 3 nodes × 3 DOFs originally
        expected_free_dofs = total_dofs - len(constrained_dofs)
        assert free_dofs == expected_free_dofs

        # Test LQR with reduced system
        Q = np.eye(2 * free_dofs)
        R = np.eye(free_dofs)

        lqr = LinearQuadraticRegulator(A, B, Q, R)
        K, _ = lqr.compute_gain_matrix()

        # Check reduced dimensions
        assert K.shape == (free_dofs, 2 * free_dofs)

        # Verify stability
        A_cl = A - B @ K
        eigenvals = np.linalg.eigvals(A_cl)
        assert np.all(np.real(eigenvals) < 0)


class TestDiscreteTimeLQR:
    """Test discrete-time LQR functionality."""

    @pytest.fixture
    def simple_discrete_system(self):
        """Create simple discrete-time system for testing."""
        # Simple 2-state discrete system
        dt = 0.01
        A_cont = np.array([[0.0, 1.0], [-1.0, -0.1]])
        B_cont = np.array([[0.0], [1.0]])

        # Discretize using ZOH
        import control as ct

        sys_cont = ct.StateSpace(A_cont, B_cont, np.eye(2), np.zeros((2, 1)))
        sys_disc = ct.sample_system(sys_cont, dt, method="zoh")

        A_disc = np.array(sys_disc.A)
        B_disc = np.array(sys_disc.B)

        return A_disc, B_disc, dt

    def test_discrete_lqr_with_statespace(self, simple_discrete_system):
        """Test discrete LQR initialization with StateSpace object."""
        A, B, dt = simple_discrete_system

        # Create discrete-time StateSpace system
        import control as ct

        sys = ct.StateSpace(A, B, np.eye(2), np.zeros((2, 1)), dt=dt)

        Q = np.eye(2)
        R = np.array([[1.0]])

        # Create LQR with sys parameter
        lqr = LinearQuadraticRegulator(sys=sys, Q=Q, R=R)

        assert lqr.is_discrete_time()
        assert np.array_equal(lqr.A, A)
        assert np.array_equal(lqr.B, B)

    def test_discrete_lqr_gain_computation(self, simple_discrete_system):
        """Test discrete-time LQR gain computation."""
        A, B, dt = simple_discrete_system

        Q = np.eye(2)
        R = np.array([[1.0]])

        lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R)
        lqr.set_discrete_time(True)

        K, S = lqr.compute_gain_matrix()

        # Check dimensions
        assert K.shape == (1, 2)
        assert S.shape == (2, 2)

        # Check that S is positive semidefinite
        eigenvals_S = np.linalg.eigvals(S)
        assert np.all(eigenvals_S >= -1e-8)

    def test_discrete_lqr_stability(self, simple_discrete_system):
        """Test that discrete LQR produces stable closed-loop system."""
        A, B, dt = simple_discrete_system

        Q = np.eye(2)
        R = np.array([[1.0]])

        lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R)
        lqr.set_discrete_time(True)

        K, _ = lqr.compute_gain_matrix()

        # Check closed-loop stability (discrete: eigenvalues inside unit circle)
        A_cl = A - B @ K
        eigenvals = np.linalg.eigvals(A_cl)
        max_magnitude = np.max(np.abs(eigenvals))

        assert max_magnitude < 1.0

    def test_cannot_change_discrete_mode_after_computation(
        self, simple_discrete_system
    ):
        """Test that discrete mode cannot be changed after gain computation."""
        A, B, dt = simple_discrete_system

        Q = np.eye(2)
        R = np.array([[1.0]])

        lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R)
        lqr.set_discrete_time(True)
        lqr.compute_gain_matrix()

        with pytest.raises(ValueError, match="Cannot change discrete/continuous mode"):
            lqr.set_discrete_time(False)

    def test_discrete_full_state_linear_initialization(self, simple_discrete_system):
        """Test FullStateLinear initialization for discrete-time control."""
        A, B, dt = simple_discrete_system

        Q = np.eye(2)
        R = np.array([[1.0]])

        lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R)
        lqr.set_discrete_time(True)
        K, _ = lqr.compute_gain_matrix()

        # Create discrete-time controller
        controller = FullStateLinear(K, B, is_discrete=True, dt=dt)

        assert controller.is_discrete
        assert controller.dt == dt

    def test_discrete_full_state_linear_requires_dt(self, simple_discrete_system):
        """Test that discrete FullStateLinear requires dt parameter."""
        A, B, dt = simple_discrete_system

        Q = np.eye(2)
        R = np.array([[1.0]])

        lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R)
        lqr.set_discrete_time(True)
        K, _ = lqr.compute_gain_matrix()

        # Should raise error without dt
        with pytest.raises(
            ValueError, match="dt must be provided for discrete-time control"
        ):
            FullStateLinear(K, B, is_discrete=True)

    def test_discrete_full_state_linear_control_computation(
        self, simple_discrete_system
    ):
        """Test discrete-time control input computation."""
        A, B, dt = simple_discrete_system

        Q = np.eye(2)
        R = np.array([[1.0]])

        lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R)
        lqr.set_discrete_time(True)
        K, _ = lqr.compute_gain_matrix()

        # Create controller
        # The discrete system is a 2-state system (not split into position/velocity)
        # For this test, we'll treat it as is without the DOF split
        controller = FullStateLinear(K, B, is_discrete=True, dt=dt)

        # Test control computation with correct state dimension
        x = np.array([1.0, 0.5])  # 2-state system
        r = np.zeros(2)  # Zero reference

        u = controller.compute_input(x, r, 0.0)

        # Should return control input (force)
        # Note: B shape is (2, 1), so ndof = 2/2 = 1
        assert u.shape == (1,)

    def test_set_discrete_time_method(self):
        """Test set_discrete_time method."""
        A = np.array([[1.0, 0.01], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        Q = np.eye(2)
        R = np.array([[1.0]])

        lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R)

        # Default should be continuous
        assert not lqr.is_discrete_time()

        # Set to discrete
        lqr.set_discrete_time(True)
        assert lqr.is_discrete_time()

        # Set back to continuous
        lqr.set_discrete_time(False)
        assert not lqr.is_discrete_time()

    def test_get_S_method(self, simple_discrete_system):
        """Test get_S method returns solution matrix."""
        A, B, dt = simple_discrete_system

        Q = np.eye(2)
        R = np.array([[1.0]])

        lqr = LinearQuadraticRegulator(A=A, B=B, Q=Q, R=R)
        lqr.set_discrete_time(True)

        S = lqr.get_S()

        # Should compute and return S
        assert S is not None
        assert S.shape == (2, 2)

        # Check that it's positive semidefinite
        eigenvals_S = np.linalg.eigvals(S)
        assert np.all(eigenvals_S >= -1e-8)
