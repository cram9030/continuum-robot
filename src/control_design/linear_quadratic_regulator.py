import numpy as np
import control as ct


class LinearQuadraticRegulator:
    """
    Linear Quadratic Regulator (LQR) controller for continuum robot beams.

    This class computes optimal control gains for linear beam systems using the
    Linear Quadratic Regulator method. It takes pre-computed stiffness and mass
    matrices from linear beam systems.

    The LQR controller minimizes the cost function:
        J = ∫(x'Qx + u'Ru)dt

    where Q is the state weighting matrix and R is the control weighting matrix.
    """

    def __init__(
        self, K_beam: np.ndarray, M_beam: np.ndarray, Q: np.ndarray, R: np.ndarray
    ):
        """
        Initialize the Linear Quadratic Regulator.

        Args:
            K_beam: Stiffness matrix from linear beam system
            M_beam: Mass matrix from linear beam system
            Q: State weighting matrix (positive semidefinite)
            R: Control weighting matrix (positive definite)

        Raises:
            ValueError: If matrix dimensions are invalid or matrices have wrong properties
        """
        self._validate_beam_matrices(K_beam, M_beam)
        self._validate_weighting_matrices(Q, R)

        self.K_beam = K_beam
        self.M_beam = M_beam
        self.Q = Q
        self.R = R
        self._A = None
        self._B = None
        self._K = None
        self._S = None
        self._E = None

    def _validate_beam_matrices(self, K_beam: np.ndarray, M_beam: np.ndarray) -> None:
        """Validate beam stiffness and mass matrices."""
        if K_beam.ndim != 2 or K_beam.shape[0] != K_beam.shape[1]:
            raise ValueError("Stiffness matrix must be square")

        if M_beam.ndim != 2 or M_beam.shape[0] != M_beam.shape[1]:
            raise ValueError("Mass matrix must be square")

        if K_beam.shape != M_beam.shape:
            raise ValueError(
                "Stiffness and mass matrices must have the same dimensions"
            )

    def _validate_weighting_matrices(self, Q: np.ndarray, R: np.ndarray) -> None:
        """Validate Q and R matrices dimensions and properties."""
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q matrix must be square")

        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise ValueError("R matrix must be square")

        # Check positive semidefinite for Q
        try:
            eigenvals_Q = np.linalg.eigvals(Q)
            if np.any(eigenvals_Q < -1e-10):  # Allow small numerical errors
                raise ValueError("Q matrix must be positive semidefinite")
        except np.linalg.LinAlgError:
            raise ValueError("Q matrix must be positive semidefinite")

        # Check positive definite for R
        try:
            eigenvals_R = np.linalg.eigvals(R)
            if np.any(eigenvals_R <= 1e-10):  # Must be strictly positive
                raise ValueError("R matrix must be positive definite")
        except np.linalg.LinAlgError:
            raise ValueError("R matrix must be positive definite")

    def get_A(self) -> np.ndarray:
        """
        Compute the A matrix for the linear system dx/dt = Ax + Bu.

        For a structural dynamics system: M*q̈ + K*q = u
        In state space form with x = [q, q̇]:
        dx/dt = [0  I ] x + [0    ] u
                [-K -C]     [M^-1]

        Returns:
            A matrix for the linearized system
        """
        if self._A is not None:
            return self._A

        # Use the provided beam matrices
        M = self.M_beam
        K = self.K_beam

        n = M.shape[0]  # Number of position DOFs

        # Create A matrix for undamped system
        # A = [0   I ]
        #     [-M^-1*K  0]
        self._A = np.zeros((2 * n, 2 * n))
        self._A[:n, n:] = np.eye(n)  # Upper right: I

        try:
            M_inv = np.linalg.inv(M)
            self._A[n:, :n] = -M_inv @ K  # Lower left: -M^-1*K
        except np.linalg.LinAlgError:
            raise ValueError("Mass matrix is singular and cannot be inverted")

        return self._A

    def get_B(self) -> np.ndarray:
        """
        Compute the B matrix for the linear system dx/dt = Ax + Bu.

        For a structural system, forces are applied directly to positions:
        B = [0    ]
            [M^-1 ]

        Returns:
            B matrix for the linearized system
        """
        if self._B is not None:
            return self._B

        # Use the provided mass matrix
        M = self.M_beam
        n = M.shape[0]  # Number of position DOFs

        # Create B matrix - assume full actuation for now
        self._B = np.zeros((2 * n, n))

        try:
            M_inv = np.linalg.inv(M)
            self._B[n:, :] = M_inv  # Lower half: M^-1
        except np.linalg.LinAlgError:
            raise ValueError("Mass matrix is singular and cannot be inverted")

        return self._B

    def compute_gain_matrix(self) -> np.ndarray:
        """
        Compute the optimal LQR gain matrix K.

        Solves the algebraic Riccati equation to find the optimal gain matrix K
        such that u = -K*x minimizes the quadratic cost function.

        Returns:
            Optimal gain matrix K

        Raises:
            ValueError: If the LQR problem cannot be solved
        """
        if self._K is not None:
            return self._K

        A = self.get_A()
        B = self.get_B()

        # Validate dimensions
        if self.Q.shape[0] != A.shape[0]:
            raise ValueError(
                f"Q matrix dimension {self.Q.shape[0]} must match state dimension {A.shape[0]}"
            )

        if self.R.shape[0] != B.shape[1]:
            raise ValueError(
                f"R matrix dimension {self.R.shape[0]} must match input dimension {B.shape[1]}"
            )

        try:
            # Solve LQR problem
            self._K, self._S, self._E = ct.lqr(A, B, self.Q, self.R)
        except Exception as e:
            raise ValueError(f"Failed to solve LQR problem: {e}")

        # Validate that the closed-loop system is stable
        A_cl = A - B @ self._K
        eigenvals = np.linalg.eigvals(A_cl)

        if np.any(np.real(eigenvals) >= 0):
            raise ValueError("LQR solution results in unstable closed-loop system")

        return self._K

    def get_K(self) -> np.ndarray:
        """
        Get the computed gain matrix K.

        Returns:
            Gain matrix K if already computed, otherwise computes it first
        """
        return self.compute_gain_matrix()
