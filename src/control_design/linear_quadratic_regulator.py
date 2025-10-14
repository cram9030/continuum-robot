import numpy as np
import control as ct


class LinearQuadraticRegulator:
    """
    Linear Quadratic Regulator (LQR) controller for continuum robot beams.

    This class computes optimal control gains for linear systems using the
    Linear Quadratic Regulator method. It takes pre-computed state-space matrices
    A and B along with weighting matrices Q and R.

    For continuous-time systems:
        The LQR controller minimizes the cost function:
        J = ∫(x'Qx + u'Ru)dt

    For discrete-time systems:
        The LQR controller minimizes the cost function:
        J = Σ(x'Qx + u'Ru)

    where Q is the state weighting matrix and R is the control weighting matrix.

    Attributes:
        A: State transition matrix
        B: Input matrix
        Q: State weighting matrix
        R: Control weighting matrix
        is_discrete: Whether the system is discrete-time
        _K: Control gain matrix (computed)
        _S: Solution to Riccati equation (computed)
        _E: Closed-loop eigenvalues (computed)
    """

    def __init__(
        self,
        A: np.ndarray = None,
        B: np.ndarray = None,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        sys: ct.StateSpace = None,
    ):
        """
        Initialize the Linear Quadratic Regulator.

        Can be initialized either with explicit matrices (A, B, Q, R) or with
        a control.StateSpace system object.

        Args:
            A: State matrix for the linear system (n_states x n_states). Required if sys is None.
            B: Input matrix for the linear system (n_states x n_inputs). Required if sys is None.
            Q: State weighting matrix (positive semidefinite). Required if sys is None.
            R: Control weighting matrix (positive definite). Required if sys is None.
            sys: Control system StateSpace object. If provided, A and B are extracted from it.

        Raises:
            ValueError: If matrix dimensions are invalid or matrices have wrong properties
            ValueError: If neither (A, B, Q, R) nor sys is provided
        """
        # Extract system matrices
        if sys is not None:
            # Extract from StateSpace system
            if not isinstance(sys, ct.StateSpace):
                raise ValueError("sys must be a control.StateSpace object")

            self.A = np.array(sys.A)
            self.B = np.array(sys.B)
            self.is_discrete = sys.dt is not None and sys.dt > 0

            # Still need Q and R to be provided
            if Q is None or R is None:
                raise ValueError(
                    "Q and R matrices must be provided even when using sys"
                )
            self.Q = Q
            self.R = R
        else:
            # All matrices must be provided
            if A is None or B is None or Q is None or R is None:
                raise ValueError("Either sys or all of (A, B, Q, R) must be provided")

            self._validate_system_matrices(A, B)
            self.A = A
            self.B = B
            self.Q = Q
            self.R = R
            self.is_discrete = False  # Default to continuous unless specified

        # Validate weighting matrices
        self._validate_weighting_matrices(self.Q, self.R)

        self._K = None
        self._S = None
        self._E = None

    def _validate_system_matrices(self, A: np.ndarray, B: np.ndarray) -> None:
        """Validate state-space matrices A and B."""
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A matrix must be square")

        if B.ndim != 2:
            raise ValueError("B matrix must be 2-dimensional")

        if A.shape[0] != B.shape[0]:
            raise ValueError(
                "A and B matrices must have compatible dimensions (A.shape[0] == B.shape[0])"
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
        Get the A matrix for the linear system dx/dt = Ax + Bu.

        Returns:
            A matrix for the linearized system
        """
        return self.A

    def get_B(self) -> np.ndarray:
        """
        Get the B matrix for the linear system dx/dt = Ax + Bu.

        Returns:
            B matrix for the linearized system
        """
        return self.B

    def compute_gain_matrix(self) -> tuple:
        """
        Compute the optimal LQR gain matrix K and solution S.

        Solves the algebraic Riccati equation to find the optimal gain matrix K
        such that u = -K*x minimizes the quadratic cost function.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (K, S)
                K: Control gain matrix (n_inputs x n_states)
                S: Solution to Riccati equation (n_states x n_states)

        Raises:
            ValueError: If the LQR problem cannot be solved
            ValueError: If the solution results in an unstable closed-loop system
        """
        if self._K is not None and self._S is not None:
            return self._K, self._S

        # Validate dimensions
        if self.Q.shape[0] != self.A.shape[0]:
            raise ValueError(
                f"Q matrix dimension {self.Q.shape[0]} must match state dimension {self.A.shape[0]}"
            )

        if self.R.shape[0] != self.B.shape[1]:
            raise ValueError(
                f"R matrix dimension {self.R.shape[0]} must match input dimension {self.B.shape[1]}"
            )

        try:
            if self.is_discrete:
                # Discrete-time LQR: dlqr(A, B, Q, R)
                self._K, self._S, self._E = ct.dlqr(self.A, self.B, self.Q, self.R)
            else:
                # Continuous-time LQR: lqr(A, B, Q, R)
                self._K, self._S, self._E = ct.lqr(self.A, self.B, self.Q, self.R)
        except Exception as e:
            raise ValueError(f"Failed to solve LQR problem: {e}")

        # Check stability of the closed-loop system using eigenvalues from control library
        if self.is_discrete:
            # Discrete-time: eigenvalues must be inside unit circle
            max_magnitude = np.max(np.abs(self._E))
            if max_magnitude >= 1.0:
                raise ValueError(
                    f"LQR solution results in unstable closed-loop system "
                    f"(max eigenvalue magnitude: {max_magnitude:.6f})"
                )
        else:
            # Continuous-time: eigenvalues must have negative real parts
            max_real_part = np.max(np.real(self._E))
            if max_real_part >= 0:
                raise ValueError(
                    f"LQR solution results in unstable closed-loop system "
                    f"(max eigenvalue real part: {max_real_part:.6f})"
                )

        return self._K, self._S

    def get_K(self) -> np.ndarray:
        """
        Get the computed control gain matrix K.

        Returns:
            Control gain matrix K if already computed, otherwise computes it first
        """
        if self._K is None:
            self.compute_gain_matrix()
        return self._K

    def get_S(self) -> np.ndarray:
        """
        Get the computed solution to Riccati equation S.

        Returns:
            Solution matrix S if already computed, otherwise computes it first
        """
        if self._S is None:
            self.compute_gain_matrix()
        return self._S

    def is_discrete_time(self) -> bool:
        """
        Check if the system is discrete-time.

        Returns:
            True if discrete-time, False if continuous-time
        """
        return self.is_discrete

    def set_discrete_time(self, is_discrete: bool):
        """
        Set whether the system is discrete-time or continuous-time.

        This should be called before compute_gain_matrix() if you need to
        override the default (continuous) or the value inferred from sys.

        Args:
            is_discrete: True for discrete-time, False for continuous-time
        """
        if self._K is not None:
            raise ValueError(
                "Cannot change discrete/continuous mode after gain has been computed"
            )
        self.is_discrete = is_discrete
