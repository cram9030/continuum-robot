"""
Tests for DiscreteKalman filter.

This module tests the discrete-time Kalman filter implementation, focusing on
Joseph form numerical stability, estimation accuracy, and edge cases.
"""

import pytest
import numpy as np

from continuum_robot.estimator.kalman_filter import DiscreteKalman


class TestDiscreteKalmanInitialization:
    """Test DiscreteKalman filter initialization."""

    @pytest.fixture
    def discrete_system_matrices(self):
        """Provide discrete-time system matrices for testing."""
        # Simple discrete-time system (sampled from continuous system with dt=0.01)
        dt = 0.01
        A_discrete = np.array([[1.0, dt], [-dt, 0.999]])  # Discretized oscillator
        B_discrete = np.array([[0.5 * dt**2], [dt]])
        C = np.array([[1.0, 0.0]])  # Measure position only
        Q = 0.01 * np.eye(2)  # Discrete process noise
        R = np.array([[0.1]])  # Measurement noise
        P = np.eye(2)  # Initial error covariance
        x0 = np.zeros(2)  # Initial state

        return A_discrete, B_discrete, C, Q, R, P, x0

    def test_successful_initialization(self, discrete_system_matrices):
        """Test successful initialization with valid matrices."""
        A, B, C, Q, R, P, x0 = discrete_system_matrices

        kf = DiscreteKalman(A, B, C, Q, R, P, x0)

        assert np.array_equal(kf.A, A)
        assert np.array_equal(kf.B, B)
        assert np.array_equal(kf.C, C)
        assert np.array_equal(kf.Q, Q)
        assert np.array_equal(kf.R, R)
        assert np.array_equal(kf.P, P)
        assert np.array_equal(kf.x_est, x0)

    def test_matrices_are_copied(self, discrete_system_matrices):
        """Test that matrices are copied, not referenced."""
        A, B, C, Q, R, P, x0 = discrete_system_matrices

        kf = DiscreteKalman(A, B, C, Q, R, P, x0)

        # Modify original matrices
        A[0, 0] = 999
        B[0, 0] = 999
        x0[0] = 999

        # Kalman filter matrices should be unchanged
        assert kf.A[0, 0] != 999
        assert kf.B[0, 0] != 999
        assert kf.x_est[0] != 999

    def test_invalid_initialization(self):
        """Test that invalid initialization parameters are caught."""
        A = np.array([[1.0, 0.1], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        C = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R_invalid = np.array([[-0.1]])  # Negative definite (invalid)
        P = np.eye(2)
        x0 = np.zeros(2)

        with pytest.raises(ValueError, match="R must be positive definite"):
            DiscreteKalman(A, B, C, Q, R_invalid, P, x0)


class TestDiscreteKalmanEstimation:
    """Test DiscreteKalman filter estimation functionality."""

    @pytest.fixture
    def initialized_filter(self):
        """Provide an initialized DiscreteKalman filter."""
        dt = 0.01
        A = np.array([[1.0, dt], [-dt, 0.999]])
        B = np.array([[0.5 * dt**2], [dt]])
        C = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R = np.array([[0.1]])
        P = np.eye(2)
        x0 = np.zeros(2)

        return DiscreteKalman(A, B, C, Q, R, P, x0)

    def test_basic_estimation_step(self, initialized_filter):
        """Test a basic estimation step with valid inputs."""
        kf = initialized_filter

        y = np.array([0.1])  # Position measurement
        u = np.array([1.0])  # Control input
        t = 0.0  # Time (unused in discrete filter)

        initial_state = kf.x_est.copy()
        estimated_state = kf.estimate_states(y, u, t)

        # Check that estimation returns correct type and shape
        assert isinstance(estimated_state, np.ndarray)
        assert estimated_state.shape == (2,)

        # Check that state has been updated
        assert not np.array_equal(estimated_state, initial_state)

    def test_multiple_estimation_steps(self, initialized_filter):
        """Test multiple consecutive estimation steps."""
        kf = initialized_filter

        states = []
        measurements = [0.0, 0.1, 0.15, 0.18, 0.2]
        inputs = [1.0, 1.0, 0.5, 0.0, -0.5]

        for i, (y_val, u_val) in enumerate(zip(measurements, inputs)):
            y = np.array([y_val])
            u = np.array([u_val])
            t = i * 0.01  # Time step

            state = kf.estimate_states(y, u, t)
            states.append(state.copy())

        # Check that we got results for all time steps
        assert len(states) == 5

        # Check that position estimate increases initially
        assert states[4][0] > states[0][0]

    def test_validation_errors_raised(self, initialized_filter):
        """Test that validation errors are properly raised."""
        kf = initialized_filter

        # Test with wrong measurement dimensions
        y_wrong = np.array([0.1, 0.2])  # Should be 1D
        u = np.array([0.0])

        with pytest.raises(ValueError, match="Measurement y must have shape"):
            kf.estimate_states(y_wrong, u, 0.0)

    def test_measurement_innovation(self, initialized_filter):
        """Test that measurement innovation properly updates state."""
        kf = initialized_filter

        # First measurement
        y1 = np.array([1.0])
        u = np.array([0.0])
        state1 = kf.estimate_states(y1, u, 0.0)

        # Second measurement with large innovation
        y2 = np.array([5.0])
        state2 = kf.estimate_states(y2, u, 0.01)

        # State should move towards new measurement
        assert state2[0] > state1[0]


class TestDiscreteKalmanJosephForm:
    """Test Joseph form numerical stability."""

    def test_joseph_form_numerical_stability(self):
        """Test that Joseph form provides numerical stability."""
        # Create system with potential numerical issues
        A = np.array([[0.9999, 0.01], [0.0, 0.9999]])
        B = np.array([[1e-4], [1e-4]])
        C = np.array([[1.0, 1.0]])  # Full state observation
        Q = 1e-6 * np.eye(2)  # Very small process noise
        R = np.array([[1e-6]])  # Very small measurement noise
        P = 1e-6 * np.eye(2)  # Very small initial covariance
        x0 = np.zeros(2)

        kf = DiscreteKalman(A, B, C, Q, R, P, x0)

        # Run multiple iterations
        for i in range(100):
            y = np.array([0.001 * i])
            u = np.array([0.0])
            kf.estimate_states(y, u, float(i))

        # Check that P remains positive semidefinite
        eigenvals = np.linalg.eigvals(kf.P)
        assert np.all(
            eigenvals > -1e-10
        ), "Covariance should remain positive semidefinite"

        # Check that P is symmetric
        assert np.allclose(kf.P, kf.P.T), "Covariance should remain symmetric"

    def test_covariance_symmetry_maintained(self):
        """Test that covariance matrix remains symmetric through updates."""
        A = np.array([[1.0, 0.1], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        C = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R = np.array([[0.1]])
        P = np.eye(2)
        x0 = np.zeros(2)

        kf = DiscreteKalman(A, B, C, Q, R, P, x0)

        # Run multiple iterations
        np.random.seed(42)
        for i in range(50):
            y = np.array([np.random.randn()])
            u = np.array([0.0])
            kf.estimate_states(y, u, i * 0.01)

            # Check symmetry after each update
            assert np.allclose(
                kf.P, kf.P.T
            ), f"Covariance not symmetric at iteration {i}"


class TestDiscreteKalmanEdgeCases:
    """Test edge cases for DiscreteKalman filter."""

    def test_zero_process_noise(self):
        """Test estimation with zero process noise."""
        A = np.array([[1.0, 0.01], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        C = np.array([[1.0, 0.0]])
        Q_zero = np.zeros((2, 2))
        R = np.array([[0.1]])
        P = np.eye(2)
        x0 = np.zeros(2)

        kf = DiscreteKalman(A, B, C, Q_zero, R, P, x0)

        y = np.array([0.5])
        u = np.array([0.0])
        result = kf.estimate_states(y, u, 0.0)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_perfect_measurements(self):
        """Test case where measurement noise is very small (near-perfect sensors)."""
        A = np.array([[1.0, 0.01], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        C = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R_small = np.array([[1e-10]])  # Very small measurement noise
        P = np.eye(2)
        x0 = np.zeros(2)

        kf = DiscreteKalman(A, B, C, Q, R_small, P, x0)

        measurement = np.array([5.0])
        u = np.array([0.0])
        result = kf.estimate_states(measurement, u, 0.0)

        # With very small R, estimate should be close to measurement
        assert abs(result[0] - measurement[0]) < 0.5

    def test_high_process_noise(self):
        """Test estimation with high process noise."""
        A = np.array([[1.0, 0.01], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        C = np.array([[1.0, 0.0]])
        Q_high = 10.0 * np.eye(2)  # High process noise
        R = np.array([[0.1]])
        P = np.eye(2)
        x0 = np.zeros(2)

        kf = DiscreteKalman(A, B, C, Q_high, R, P, x0)

        initial_trace = np.trace(P)
        y = np.array([0.5])
        u = np.array([1.0])
        result = kf.estimate_states(y, u, 0.0)

        assert isinstance(result, np.ndarray)
        # High process noise should increase uncertainty
        assert np.trace(kf.P) > initial_trace

    def test_covariance_convergence(self):
        """Test that covariance converges over time."""
        A = np.array([[1.0, 0.01], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        C = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R = np.array([[0.1]])
        P = np.eye(2)
        x0 = np.zeros(2)

        kf = DiscreteKalman(A, B, C, Q, R, P, x0)

        initial_trace = np.trace(kf.P)

        # Run many iterations with consistent measurements
        np.random.seed(42)
        for i in range(50):
            y = np.array([1.0 + 0.05 * np.random.randn()])
            u = np.array([0.0])
            kf.estimate_states(y, u, i * 0.01)

        final_trace = np.trace(kf.P)

        # Covariance should decrease (more confident estimates)
        assert final_trace < initial_trace


class TestDiscreteKalmanAnalytical:
    """Test DiscreteKalman against known analytical solutions."""

    def test_comparison_with_known_solution(self):
        """Test discrete Kalman filter against known analytical solution."""
        # Simple 1D system: x[k+1] = x[k], y[k] = x[k]
        A = np.array([[1.0]])
        B = np.array([[0.0]])
        C = np.array([[1.0]])
        Q = np.array([[0.1]])
        R = np.array([[0.1]])
        P = np.array([[1.0]])
        x0 = np.array([0.0])

        kf = DiscreteKalman(A, B, C, Q, R, P, x0)

        # Known measurement
        y = np.array([5.0])
        u = np.array([0.0])

        # Analytical Kalman gain: K = P_pred / (P_pred + R)
        # With prediction: P_pred = A*P*A' + Q = 1*1*1 + 0.1 = 1.1
        # K = 1.1 / (1.1 + 0.1) = 1.1 / 1.2 ≈ 0.9167
        # Updated state: x = 0 + K * (5 - 0) ≈ 4.583
        result = kf.estimate_states(y, u, 0.0)

        # Joseph form may have slight numerical differences from standard form
        expected_state = 0.0 + (1.1 / (1.1 + 0.1)) * 5.0
        assert np.abs(result[0] - expected_state) < 0.05

    def test_steady_state_kalman_gain(self):
        """Test that Kalman gain converges to steady state."""
        # Simple 1D system
        A = np.array([[1.0]])
        B = np.array([[0.0]])
        C = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[0.1]])
        P = np.array([[10.0]])  # Start with high uncertainty
        x0 = np.array([0.0])

        kf = DiscreteKalman(A, B, C, Q, R, P, x0)

        # Run many iterations to reach steady state
        gains = []
        for i in range(100):
            y = np.array([1.0])
            u = np.array([0.0])

            # Calculate Kalman gain before update
            P_pred = A @ kf.P @ A.T + Q
            S = C @ P_pred @ C.T + R
            K = P_pred @ C.T @ np.linalg.inv(S)
            gains.append(K[0, 0])

            kf.estimate_states(y, u, i * 0.01)

        # Check that gain converges (variance in last 20 iterations is small)
        recent_gains = np.array(gains[-20:])
        assert (
            np.var(recent_gains) < 1e-6
        ), "Kalman gain should converge to steady state"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
