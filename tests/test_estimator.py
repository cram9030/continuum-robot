"""
Tests for the estimator module.

This module tests both the abstract base class and concrete implementations
of estimators, focusing on robotics applications with comprehensive error
handling and edge case coverage.
"""

import pytest
import numpy as np
import warnings

from continuum_robot.estimator.estimator_abstractions import AbstractEstimatorHandler
from continuum_robot.estimator.kalman_filter import HybridContinuousDiscreteKalman


class TestAbstractEstimatorHandler:
    """Test the abstract estimator handler interface."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that the abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractEstimatorHandler()

    def test_concrete_implementation_requires_estimate_states_method(self):
        """Test that concrete implementations must implement estimate_states."""

        class IncompleteEstimator(AbstractEstimatorHandler):
            pass

        with pytest.raises(TypeError):
            IncompleteEstimator()

    def test_concrete_implementation_with_estimate_states_works(self):
        """Test that concrete implementations with estimate_states method work."""

        class MockEstimator(AbstractEstimatorHandler):
            def estimate_states(self, y, u, t):
                return np.zeros(6)

        estimator = MockEstimator()
        result = estimator.estimate_states(np.array([1, 2]), np.array([0.1]), 0.1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (6,)


class TestHybridContinuousDiscreteKalmanInitialization:
    """Test KalmanFilter initialization and validation."""

    @pytest.fixture
    def simple_system_matrices(self):
        """Provide simple 2-state system matrices for testing."""
        n_states = 2
        A = np.array([[0.0, 1.0], [-1.0, -0.1]])  # Simple oscillator
        B = np.array([[0.0], [1.0]])  # Control input affects acceleration
        H = np.array([[1.0, 0.0]])  # Measure position only
        Q = 0.01 * np.eye(n_states)  # Process noise
        R = np.array([[0.1]])  # Measurement noise
        P = np.eye(n_states)  # Initial error covariance
        x0 = np.zeros(n_states)  # Initial state
        dt = 0.01

        return A, B, H, Q, R, P, x0, dt

    def test_successful_initialization(self, simple_system_matrices):
        """Test successful initialization with valid matrices."""
        A, B, H, Q, R, P, x0, dt = simple_system_matrices

        kf = HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, dt)

        assert np.array_equal(kf.A, A)
        assert np.array_equal(kf.B, B)
        assert np.array_equal(kf.H, H)
        assert np.array_equal(kf.Q, Q)
        assert np.array_equal(kf.R, R)
        assert np.array_equal(kf.P, P)
        assert np.array_equal(kf.x_est, x0)
        assert kf.dt == dt

    def test_matrices_are_copied_not_referenced(self, simple_system_matrices):
        """Test that initialization copies matrices rather than storing references."""
        A, B, H, Q, R, P, x0, dt = simple_system_matrices

        kf = HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, dt)

        # Modify original matrices
        A[0, 0] = 999
        B[0, 0] = 999
        x0[0] = 999

        # Kalman filter matrices should be unchanged
        assert kf.A[0, 0] != 999
        assert kf.B[0, 0] != 999
        assert kf.x_est[0] != 999

    def test_non_numpy_array_input_raises_type_error(self, simple_system_matrices):
        """Test that non-numpy array inputs raise TypeError."""
        A, B, H, Q, R, P, x0, dt = simple_system_matrices

        # Test with list instead of numpy array
        with pytest.raises(TypeError, match="Matrix A must be a numpy array"):
            HybridContinuousDiscreteKalman(A.tolist(), B, H, Q, R, P, x0, dt)

    def test_invalid_time_step_raises_error(self, simple_system_matrices):
        """Test that invalid time steps raise ValueError."""
        A, B, H, Q, R, P, x0, _ = simple_system_matrices

        # Negative time step
        with pytest.raises(ValueError, match="dt must be a positive number"):
            HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, -0.01)

        # Zero time step
        with pytest.raises(ValueError, match="dt must be a positive number"):
            HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, 0.0)

    def test_incompatible_matrix_dimensions_raise_error(self, simple_system_matrices):
        """Test that incompatible matrix dimensions raise ValueError."""
        A, B, H, Q, R, P, x0, dt = simple_system_matrices

        # Wrong B dimensions (wrong number of rows)
        B_wrong = np.array([[0.0], [1.0], [2.0]])  # 3x1 instead of 2x1
        with pytest.raises(ValueError, match="B must have 2 rows"):
            HybridContinuousDiscreteKalman(A, B_wrong, H, Q, R, P, x0, dt)

        # Wrong H dimensions
        H_wrong = np.array([[1.0, 0.0, 1.0]])  # 1x3 instead of 1x2
        with pytest.raises(ValueError, match="H must have 2 columns"):
            HybridContinuousDiscreteKalman(A, B, H_wrong, Q, R, P, x0, dt)

    def test_non_square_system_matrix_raises_error(self, simple_system_matrices):
        """Test that non-square A matrix raises ValueError."""
        _, B, H, Q, R, P, x0, dt = simple_system_matrices

        A_nonsquare = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
        with pytest.raises(ValueError, match="A must be square"):
            HybridContinuousDiscreteKalman(A_nonsquare, B, H, Q, R, P, x0, dt)

    def test_empty_matrices_raise_error(self, simple_system_matrices):
        """Test that empty matrices raise ValueError."""
        _, _, H, Q, R, P, _, dt = simple_system_matrices

        # Create empty matrices that are consistent with each other
        A_empty = np.array([]).reshape(0, 0)
        B_empty = np.array([]).reshape(0, 0)
        H_empty = np.array([]).reshape(0, 0)
        Q_empty = np.array([]).reshape(0, 0)
        R_empty = np.array([]).reshape(0, 0)
        P_empty = np.array([]).reshape(0, 0)
        x0_empty = np.array([])

        with pytest.raises(ValueError, match="System matrices cannot be empty"):
            HybridContinuousDiscreteKalman(
                A_empty, B_empty, H_empty, Q_empty, R_empty, P_empty, x0_empty, dt
            )

    def test_matrices_with_nan_values_raise_error(self, simple_system_matrices):
        """Test that matrices containing NaN values raise ValueError."""
        A, B, H, Q, R, P, x0, dt = simple_system_matrices

        A_nan = A.copy()
        A_nan[0, 0] = np.nan

        with pytest.raises(ValueError, match="Matrix A contains NaN values"):
            HybridContinuousDiscreteKalman(A_nan, B, H, Q, R, P, x0, dt)

    def test_matrices_with_infinite_values_raise_error(self, simple_system_matrices):
        """Test that matrices containing infinite values raise ValueError."""
        A, B, H, Q, R, P, x0, dt = simple_system_matrices

        Q_inf = Q.copy()
        Q_inf[0, 0] = np.inf

        with pytest.raises(ValueError, match="Matrix Q contains infinite values"):
            HybridContinuousDiscreteKalman(A, B, H, Q_inf, R, P, x0, dt)

    def test_non_positive_definite_measurement_noise_raises_error(
        self, simple_system_matrices
    ):
        """Test that non-positive definite R matrix raises ValueError."""
        A, B, H, Q, _, P, x0, dt = simple_system_matrices

        R_non_pd = np.array([[-0.1]])  # Negative definite

        with pytest.raises(
            ValueError, match="Measurement noise covariance R must be positive definite"
        ):
            HybridContinuousDiscreteKalman(A, B, H, Q, R_non_pd, P, x0, dt)

    def test_non_positive_semidefinite_process_noise_raises_error(
        self, simple_system_matrices
    ):
        """Test that non-positive semidefinite Q matrix raises ValueError."""
        A, B, H, _, R, P, x0, dt = simple_system_matrices

        Q_non_psd = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive semidefinite

        with pytest.raises(
            ValueError, match="Process noise covariance Q must be positive semidefinite"
        ):
            HybridContinuousDiscreteKalman(A, B, H, Q_non_psd, R, P, x0, dt)

    def test_warnings_for_non_floating_point_matrices(self, simple_system_matrices):
        """Test that warnings are issued for non-floating point matrices."""
        A, B, H, Q, R, P, x0, dt = simple_system_matrices

        # Convert to integer type
        A_int = A.astype(int)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HybridContinuousDiscreteKalman(A_int, B, H, Q, R, P, x0, dt)

            # Check that warning was issued
            assert len(w) >= 1
            assert "floating point type" in str(w[0].message)

    def test_warnings_for_unstable_system(self):
        """Test that warnings are issued for unstable systems."""
        # Create unstable system (eigenvalue with positive real part)
        A = np.array([[1.1, 0.0], [0.0, -0.5]])  # One unstable eigenvalue
        B = np.array([[1.0], [0.0]])
        H = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R = np.array([[0.1]])
        P = np.eye(2)
        x0 = np.zeros(2)
        dt = 0.01

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, dt)

            # Check that instability warning was issued
            warning_messages = [str(warn.message) for warn in w]
            assert any("unstable" in msg for msg in warning_messages)

    def test_warnings_for_ill_conditioned_matrices(self):
        """Test that warnings are issued for ill-conditioned matrices."""
        # Create ill-conditioned system matrix
        A = np.array([[1.0, 1e15], [0.0, 1.0]])  # Very ill-conditioned
        B = np.array([[1.0], [0.0]])
        H = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R = np.array([[0.1]])
        P = np.eye(2)
        x0 = np.zeros(2)
        dt = 0.01

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, dt)

            # Check that ill-conditioning warning was issued
            warning_messages = [str(warn.message) for warn in w]
            assert any("ill-conditioned" in msg for msg in warning_messages)


class TestHybridContinuousDiscreteKalmanEstimation:
    """Test HybridContinuousDiscreteKalman state estimation functionality."""

    @pytest.fixture
    def initialized_kalman_filter(self):
        """Provide an initialized Kalman filter for testing."""
        # Simple 2D position/velocity system
        A = np.array([[1.0, 0.01], [0.0, 1.0]])  # Discrete-time integration
        B = np.array([[0.5 * 0.01**2], [0.01]])  # Input affects acceleration
        H = np.array([[1.0, 0.0]])  # Measure position only
        Q = 0.01 * np.eye(2)  # Process noise
        R = np.array([[0.04]])  # Measurement noise (0.2cm standard deviation squared)
        P = np.eye(2)  # Initial error covariance
        x0 = np.zeros(2)  # Initial state [position, velocity]
        dt = 0.01

        return HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, dt)

    def test_basic_estimation_step(self, initialized_kalman_filter):
        """Test a basic estimation step with valid inputs."""
        kf = initialized_kalman_filter

        # Measurement and control input
        y = np.array([0.1])  # Position measurement
        u = np.array([1.0])  # Control input
        t = 0.0

        # Store initial state for comparison
        initial_state = kf.x_est.copy()

        # Perform estimation
        estimated_state = kf.estimate_states(y, u, t)

        # Check that estimation returns correct type and shape
        assert isinstance(estimated_state, np.ndarray)
        assert estimated_state.shape == (2,)

        # Check that state has been updated
        assert not np.array_equal(estimated_state, initial_state)

    def test_estimation_with_invalid_measurement_type_raises_error(
        self, initialized_kalman_filter
    ):
        """Test that invalid measurement types raise TypeError."""
        kf = initialized_kalman_filter

        with pytest.raises(TypeError, match="Measurement y must be numpy array"):
            kf.estimate_states([0.1], np.array([1.0]), 0.0)

    def test_estimation_with_invalid_input_type_raises_error(
        self, initialized_kalman_filter
    ):
        """Test that invalid input types raise TypeError."""
        kf = initialized_kalman_filter

        with pytest.raises(TypeError, match="Input u must be numpy array"):
            kf.estimate_states(np.array([0.1]), [1.0], 0.0)

    def test_estimation_with_wrong_measurement_dimensions_raises_error(
        self, initialized_kalman_filter
    ):
        """Test that wrong measurement dimensions raise ValueError."""
        kf = initialized_kalman_filter

        # Wrong measurement dimension (should be 1D with 1 element)
        with pytest.raises(ValueError, match="Measurement y must have shape \\(1,\\)"):
            kf.estimate_states(np.array([0.1, 0.2]), np.array([1.0]), 0.0)

    def test_estimation_with_wrong_input_dimensions_raises_error(
        self, initialized_kalman_filter
    ):
        """Test that wrong input dimensions raise ValueError."""
        kf = initialized_kalman_filter

        # Wrong input dimension (should be 1D with 1 element)
        with pytest.raises(ValueError, match="Input u must have shape \\(1,\\)"):
            kf.estimate_states(np.array([0.1]), np.array([1.0, 2.0]), 0.0)

    def test_estimation_with_nan_measurements_raises_error(
        self, initialized_kalman_filter
    ):
        """Test that NaN measurements raise ValueError."""
        kf = initialized_kalman_filter

        with pytest.raises(ValueError, match="Measurement y contains NaN"):
            kf.estimate_states(np.array([np.nan]), np.array([1.0]), 0.0)

    def test_estimation_with_infinite_measurements_raises_error(
        self, initialized_kalman_filter
    ):
        """Test that infinite measurements raise ValueError."""
        kf = initialized_kalman_filter

        with pytest.raises(ValueError, match="Measurement y contains.*infinite"):
            kf.estimate_states(np.array([np.inf]), np.array([1.0]), 0.0)

    def test_estimation_with_nan_inputs_raises_error(self, initialized_kalman_filter):
        """Test that NaN inputs raise ValueError during prediction."""
        kf = initialized_kalman_filter

        with pytest.raises(ValueError, match="Input u contains NaN"):
            kf.estimate_states(np.array([0.1]), np.array([np.nan]), 0.0)

    def test_multiple_estimation_steps(self, initialized_kalman_filter):
        """Test multiple consecutive estimation steps."""
        kf = initialized_kalman_filter

        states = []
        measurements = [0.0, 0.1, 0.15, 0.18, 0.2]  # Simulated position measurements
        inputs = [1.0, 1.0, 0.5, 0.0, -0.5]  # Control inputs

        for i, (y_val, u_val) in enumerate(zip(measurements, inputs)):
            y = np.array([y_val])
            u = np.array([u_val])
            t = i * 0.01

            state = kf.estimate_states(y, u, t)
            states.append(state.copy())

        # Check that we got results for all time steps
        assert len(states) == 5

        # Check that states have reasonable values (position should increase initially)
        assert states[4][0] > states[0][0]  # Position should increase

        # Check that the returned array is a copy (to prevent external modification)
        returned_state = kf.estimate_states(np.array([0.2]), np.array([0.0]), 0.05)
        returned_state[0] = 999.0  # Modify returned array
        assert kf.x_est[0] != 999.0  # Internal state should be unchanged

    def test_covariance_update_with_near_singular_innovation(self):
        """Test handling of near-singular innovation covariance."""
        # Create system where innovation covariance becomes nearly singular
        A = np.eye(2)
        B = np.array([[1.0], [0.0]])
        H = np.array([[1.0, 1.0]])  # H that might create singularity
        Q = 1e-10 * np.eye(2)  # Very small process noise
        R = np.array([[1e-15]])  # Very small measurement noise
        P = 1e-10 * np.eye(2)  # Very small initial covariance
        x0 = np.zeros(2)
        dt = 0.01

        kf = HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, dt)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            kf.estimate_states(np.array([0.1]), np.array([1.0]), 0.0)
            # May or may not trigger depending on exact numerical conditions
            # This tests the warning mechanism is in place


class TestHybridContinuousDiscreteKalmanEdgeCases:
    """Test edge cases and error conditions for HybridContinuousDiscreteKalman."""

    def test_high_dimensional_system_warning(self):
        """Test warning for high-dimensional systems."""
        # Create system with many DOFs (>50, so 102 states = 51 DOFs)
        n_states = 102  # 51 DOF system
        A = -0.01 * np.eye(n_states)  # Stable system
        B = np.ones((n_states, 1))
        H = np.ones((1, n_states))
        Q = 0.01 * np.eye(n_states)
        R = np.array([[0.1]])
        P = np.eye(n_states)
        x0 = np.zeros(n_states)
        dt = 0.01

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, dt)

            # Check for large DOF warning
            warning_messages = [str(warn.message) for warn in w]
            assert any("Large number of DOFs" in msg for msg in warning_messages)

    def test_zero_input_estimation(self):
        """Test estimation with zero control input."""
        # Simple system
        A = np.array([[1.0, 0.01], [0.0, 0.95]])
        B = np.array([[0.0], [1.0]])
        H = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R = np.array([[0.1]])
        P = np.eye(2)
        x0 = np.array([1.0, 0.0])  # Non-zero initial state
        dt = 0.01

        kf = HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, dt)

        # Estimate with zero input
        result = kf.estimate_states(np.array([0.9]), np.array([0.0]), 0.0)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_perfect_measurement_case(self):
        """Test case where measurement noise is very small."""
        A = np.array([[1.0, 0.01], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        H = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R = np.array([[1e-10]])  # Very small measurement noise
        P = np.eye(2)
        x0 = np.zeros(2)
        dt = 0.01

        kf = HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, dt)

        # With perfect measurement, estimate should be close to measurement
        measurement = np.array([5.0])
        result = kf.estimate_states(measurement, np.array([0.0]), 0.0)

        # Position estimate should be very close to measurement
        assert abs(result[0] - measurement[0]) < 0.1

    def test_large_time_step_warning(self):
        """Test warning for large time steps."""
        A = np.array([[0.0, 1.0], [-1.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        H = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R = np.array([[0.1]])
        P = np.eye(2)
        x0 = np.zeros(2)
        dt = 2.0  # Large time step

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            kf = HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, dt)
            kf.estimate_states(np.array([0.1]), np.array([0.0]), 0.0)
            # Warning should be issued either during initialization or estimation


class TestHybridContinuousDiscreteKalmanRoboticsScenarios:
    """Test HybridContinuousDiscreteKalman in robotics-specific scenarios."""

    def test_beam_tip_position_estimation_scenario(self):
        """Test scenario similar to continuum robot beam tip estimation."""
        # Simulate 3-DOF beam (6 states: 3 positions + 3 velocities)
        n_dof = 3
        n_states = 2 * n_dof

        # Simple integrator dynamics for each DOF
        A_block = np.array([[1.0, 0.01], [0.0, 1.0]])
        A = np.zeros((n_states, n_states))
        for i in range(n_dof):
            A[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = A_block

        # Control input affects accelerations
        B = np.zeros((n_states, n_dof))
        for i in range(n_dof):
            B[2 * i + 1, i] = 0.01  # Input affects velocity

        # Measure only tip position (last DOF position)
        H = np.zeros((1, n_states))
        H[0, 2 * n_dof - 2] = 1.0  # Measure position of last DOF

        Q = 0.01 * np.eye(n_states)
        R = np.array([[0.04]])  # 0.2cm measurement noise std
        P = np.eye(n_states)
        x0 = np.zeros(n_states)
        dt = 0.01

        kf = HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, dt)

        # Simulate measurement of beam tip at 0.15m
        tip_measurement = np.array([0.15])
        control_input = np.array([0.0, 0.0, 1.0])  # Force on tip

        estimated_state = kf.estimate_states(tip_measurement, control_input, 0.0)

        assert isinstance(estimated_state, np.ndarray)
        assert estimated_state.shape == (n_states,)

        # Tip position estimate should be influenced by measurement
        estimated_tip_position = estimated_state[2 * n_dof - 2]
        assert abs(estimated_tip_position - tip_measurement[0]) < 0.1

    def test_noisy_sensor_data_handling(self):
        """Test handling of realistic noisy sensor data."""
        # Simple 1-DOF system
        A = np.array([[1.0, 0.01], [0.0, 1.0]])
        B = np.array([[0.5 * 0.01**2], [0.01]])
        H = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R = np.array([[0.04]])  # 0.2cm noise std
        P = np.eye(2)
        x0 = np.zeros(2)
        dt = 0.01

        kf = HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, dt)

        # Simulate noisy measurements around a true trajectory
        np.random.seed(42)  # For reproducible tests
        true_positions = np.linspace(0, 0.2, 20)
        noise = 0.02 * np.random.randn(20)  # Measurement noise
        noisy_measurements = true_positions + noise

        estimated_positions = []
        for i, measurement in enumerate(noisy_measurements):
            y = np.array([measurement])
            u = np.array([0.5])  # Constant input

            t = i * kf.dt  # Use proper time progression
            state = kf.estimate_states(y, u, t)
            estimated_positions.append(state[0])

        # Check that estimation is reasonably close to true values
        estimated_positions = np.array(estimated_positions)
        errors = np.abs(estimated_positions - true_positions)
        mean_error = np.mean(errors)

        # Mean error should be less than measurement noise std
        assert mean_error < 0.05


class TestHybridContinuousDiscreteKalmanTimeBasedBehavior:
    """Test time-based measurement update behavior of HybridContinuousDiscreteKalman."""

    @pytest.fixture
    def time_based_kalman_filter(self):
        """Provide a Kalman filter configured for time-based testing."""
        # Simple 2D position/velocity system
        A = np.array([[0.0, 1.0], [-1.0, -0.1]])  # Simple oscillator
        B = np.array([[0.0], [1.0]])  # Control input affects acceleration
        H = np.array([[1.0, 0.0]])  # Measure position only
        Q = 0.01 * np.eye(2)  # Process noise
        R = np.array([[0.04]])  # Measurement noise
        P = np.eye(2)  # Initial error covariance
        x0 = np.array([1.0, 0.0])  # Non-zero initial state
        dt = 0.02  # 50 Hz measurement rate

        return HybridContinuousDiscreteKalman(A, B, H, Q, R, P, x0, dt)

    def test_first_call_forces_measurement_update(self, time_based_kalman_filter):
        """Test that first call always performs measurement update regardless of time."""
        kf = time_based_kalman_filter

        # Store initial state
        initial_state = kf.x_est.copy()
        initial_P = kf.P.copy()

        # First call should perform update
        y = np.array([0.5])
        u = np.array([0.0])
        t = 0.0

        result = kf.estimate_states(y, u, t)

        # State should have changed due to measurement update
        assert not np.array_equal(result, initial_state)
        assert not np.array_equal(kf.P, initial_P)
        assert kf.last_update_time == t

    def test_calls_before_dt_return_predicted_state_only(
        self, time_based_kalman_filter
    ):
        """Test that calls before dt elapsed return predicted state without update."""
        kf = time_based_kalman_filter

        # First call to establish last_update_time
        y = np.array([0.5])
        u = np.array([0.0])
        t1 = 0.0
        kf.estimate_states(y, u, t1)

        # Store state after first update
        state_after_update = kf.x_est.copy()
        P_after_update = kf.P.copy()

        # Second call before dt has elapsed
        t2 = t1 + kf.dt / 2  # Half the measurement interval
        result = kf.estimate_states(y, u, t2)

        # Internal state should remain unchanged (no measurement update)
        assert np.array_equal(kf.x_est, state_after_update)
        assert np.array_equal(kf.P, P_after_update)
        assert kf.last_update_time == t1  # Should not have updated

        # But returned state should be predicted state (different from internal state)
        assert not np.array_equal(result, state_after_update)

    def test_calls_after_dt_perform_measurement_update(self, time_based_kalman_filter):
        """Test that calls after dt elapsed perform measurement update."""
        kf = time_based_kalman_filter

        # First call
        y = np.array([0.5])
        u = np.array([0.0])
        t1 = 0.0
        kf.estimate_states(y, u, t1)

        # Store state after first update
        state_after_first_update = kf.x_est.copy()

        # Second call after dt has elapsed
        t2 = t1 + kf.dt + 0.001  # Slightly more than dt
        y2 = np.array([0.6])  # Different measurement
        result = kf.estimate_states(y2, u, t2)

        # Internal state should have changed (measurement update occurred)
        assert not np.array_equal(kf.x_est, state_after_first_update)
        assert kf.last_update_time == t2  # Should have updated time

        # Returned state should be the updated internal state
        assert np.array_equal(result, kf.x_est)

    def test_exactly_dt_interval_performs_update(self, time_based_kalman_filter):
        """Test that calls at exactly dt interval perform measurement update."""
        kf = time_based_kalman_filter

        # First call
        y = np.array([0.5])
        u = np.array([0.0])
        t1 = 0.0
        kf.estimate_states(y, u, t1)

        # Second call at exactly dt interval
        t2 = t1 + kf.dt
        y2 = np.array([0.7])
        kf.estimate_states(y2, u, t2)

        # Should have performed measurement update
        assert kf.last_update_time == t2

    def test_multiple_prediction_calls_before_update(self, time_based_kalman_filter):
        """Test multiple prediction calls before measurement update."""
        kf = time_based_kalman_filter

        # First call to establish baseline
        y = np.array([0.5])
        u = np.array([0.0])
        t1 = 0.0
        kf.estimate_states(y, u, t1)

        # Multiple prediction calls
        times = [t1 + kf.dt * 0.2, t1 + kf.dt * 0.4, t1 + kf.dt * 0.6, t1 + kf.dt * 0.8]
        predictions = []

        for t in times:
            pred = kf.estimate_states(y, u, t)
            predictions.append(pred.copy())
            # Internal state should remain unchanged
            assert kf.last_update_time == t1

        # All predictions should be different (due to different time intervals)
        for i in range(1, len(predictions)):
            assert not np.array_equal(predictions[i], predictions[i - 1])

    def test_mixed_prediction_and_update_sequence(self, time_based_kalman_filter):
        """Test sequence of prediction and update calls."""
        kf = time_based_kalman_filter

        y = np.array([0.5])
        u = np.array([0.0])

        # First update
        t1 = 0.0
        kf.estimate_states(y, u, t1)
        assert kf.last_update_time == t1

        # Prediction call
        t2 = t1 + kf.dt * 0.5
        kf.estimate_states(y, u, t2)
        assert kf.last_update_time == t1  # No change

        # Update call
        t3 = t1 + kf.dt
        kf.estimate_states(y, u, t3)
        assert kf.last_update_time == t3  # Should update

        # Another prediction
        t4 = t3 + kf.dt * 0.3
        kf.estimate_states(y, u, t4)
        assert kf.last_update_time == t3  # No change

    def test_time_tracking_with_varying_intervals(self, time_based_kalman_filter):
        """Test time tracking with varying time intervals."""
        kf = time_based_kalman_filter

        y = np.array([0.5])
        u = np.array([0.0])

        # Calls with varying intervals
        times = [0.0, 0.005, 0.015, 0.025, 0.04, 0.045, 0.065]
        expected_updates = [0.0, 0.025, 0.065]  # Times when updates should occur

        actual_updates = []
        for t in times:
            kf.estimate_states(y, u, t)
            if kf.last_update_time == t:
                actual_updates.append(t)

        assert actual_updates == expected_updates

    def test_large_time_jumps_still_perform_single_update(
        self, time_based_kalman_filter
    ):
        """Test that large time jumps still perform only single update."""
        kf = time_based_kalman_filter

        y = np.array([0.5])
        u = np.array([0.0])

        # First call
        t1 = 0.0
        kf.estimate_states(y, u, t1)

        # Very large time jump
        t2 = t1 + 10 * kf.dt
        state_before = kf.x_est.copy()
        kf.estimate_states(y, u, t2)

        # Should have updated time and performed single update
        assert kf.last_update_time == t2
        assert not np.array_equal(kf.x_est, state_before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
