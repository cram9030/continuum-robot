"""
Discretization utilities for continuous-time linear systems.

This module provides functions to convert continuous-time state-space models
to discrete-time using various methods, primarily zero-order hold (ZOH).
"""

import numpy as np
from scipy.linalg import expm


def continuous_to_discrete_zoh(
    A: np.ndarray, B: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert continuous-time system to discrete-time using zero-order hold.

    Given a continuous-time linear system:
        dx/dt = A @ x + B @ u

    Compute the equivalent discrete-time system:
        x[k+1] = A_d @ x[k] + B_d @ u[k]

    where:
        A_d = expm(A * dt)
        B_d = integral(expm(A * tau) @ B, 0, dt) = inv(A) @ (A_d - I) @ B

    For singular or near-singular A, uses first-order approximation:
        B_d ≈ B * dt

    Args:
        A: Continuous-time state transition matrix (n_states x n_states)
        B: Continuous-time control input matrix (n_states x n_inputs)
        dt: Sampling time (must be positive)

    Returns:
        Tuple of (A_discrete, B_discrete):
            A_discrete: Discrete-time state transition matrix (n_states x n_states)
            B_discrete: Discrete-time control input matrix (n_states x n_inputs)

    Raises:
        ValueError: If inputs are invalid (wrong dimensions, non-positive dt)
    """
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise TypeError("A and B must be numpy arrays")

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square matrix, got shape {A.shape}")

    n_states = A.shape[0]
    if B.shape[0] != n_states:
        raise ValueError(f"B must have {n_states} rows to match A, got {B.shape[0]}")

    if not isinstance(dt, (int, float)) or dt <= 0:
        raise ValueError(f"dt must be a positive number, got {dt}")

    # Compute discrete-time state transition matrix
    A_discrete = expm(A * dt)

    # Compute discrete-time control input matrix
    # Check if A is full rank
    rank_A = np.linalg.matrix_rank(A)

    if rank_A == n_states:
        # A is full rank, use exact formula: inv(A) @ (expm(A*dt) - I) @ B
        try:
            B_discrete = np.linalg.solve(A, (A_discrete - np.eye(n_states)) @ B)
        except np.linalg.LinAlgError:
            # Fallback to first-order approximation if solve fails
            B_discrete = B * dt
    else:
        # A is singular or near-singular, use first-order approximation
        B_discrete = B * dt

    return A_discrete, B_discrete


def discretize_covariance_matrix(Q_continuous: np.ndarray, dt: float) -> np.ndarray:
    """
    Convert continuous-time process noise covariance to discrete-time.

    For continuous-time process noise Q_continuous, the discrete-time equivalent is:
        Q_discrete ≈ Q_continuous * dt

    This is a first-order approximation. For more accurate results, especially with
    large dt or highly dynamic systems, consider using Van Loan's method.

    Args:
        Q_continuous: Continuous-time process noise covariance (n_states x n_states)
        dt: Sampling time (must be positive)

    Returns:
        Q_discrete: Discrete-time process noise covariance (n_states x n_states)

    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(Q_continuous, np.ndarray):
        raise TypeError("Q_continuous must be a numpy array")

    if Q_continuous.ndim != 2 or Q_continuous.shape[0] != Q_continuous.shape[1]:
        raise ValueError(
            f"Q_continuous must be square matrix, got shape {Q_continuous.shape}"
        )

    if not isinstance(dt, (int, float)) or dt <= 0:
        raise ValueError(f"dt must be a positive number, got {dt}")

    return Q_continuous * dt


def discretize_system_with_covariance(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Discretize a continuous-time LTI system including process noise covariance.

    Convenience function that applies zero-order hold discretization to both
    the system matrices and the process noise covariance.

    Args:
        A: Continuous-time state transition matrix (n_states x n_states)
        B: Continuous-time control input matrix (n_states x n_inputs)
        Q: Continuous-time process noise covariance (n_states x n_states)
        dt: Sampling time (must be positive)

    Returns:
        Tuple of (A_discrete, B_discrete, Q_discrete)

    Raises:
        ValueError: If inputs are invalid
    """
    A_d, B_d = continuous_to_discrete_zoh(A, B, dt)
    Q_d = discretize_covariance_matrix(Q, dt)

    return A_d, B_d, Q_d
