from typing import List, Callable
import numpy as np
from .abstractions import AbstractForce, AbstractInputHandler


class ForceRegistry:
    """Manages force components independently from beam dynamics."""

    def __init__(self):
        """Initialize empty force registry."""
        self._forces = []

    def register(self, force_instance: AbstractForce) -> None:
        """
        Register a force component if it's enabled.

        Args:
            force_instance: Instance of a force component implementing AbstractForce
        """
        if force_instance.is_enabled():
            self._forces.append(force_instance)

    def unregister(self, force_instance: AbstractForce) -> bool:
        """
        Unregister a force component.

        Args:
            force_instance: Force instance to remove

        Returns:
            True if force was found and removed, False otherwise
        """
        if force_instance in self._forces:
            self._forces.remove(force_instance)
            return True
        return False

    def clear(self) -> None:
        """Remove all registered force components."""
        self._forces.clear()

    def get_registered_forces(self) -> List[AbstractForce]:
        """
        Get list of registered force components.

        Returns:
            List of registered force instances
        """
        return self._forces.copy()

    def create_aggregated_function(self) -> Callable:
        """
        Create function that computes total forces from all registered components.

        Returns:
            Function that computes total forces: forces(x, t) -> np.ndarray
        """

        def aggregate_forces(x: np.ndarray, t: float = 0.0) -> np.ndarray:
            """Compute combined forces from all registered components."""
            if not self._forces:
                n_states = len(x) // 2
                return np.zeros(n_states)

            total_forces = None
            for force in self._forces:
                if force.is_enabled():
                    force_contrib = force.compute_forces(x, t)
                    if total_forces is None:
                        total_forces = force_contrib.copy()
                    else:
                        total_forces += force_contrib

            # Return zero forces if no enabled components
            if total_forces is None:
                n_states = len(x) // 2
                return np.zeros(n_states)

            return total_forces

        return aggregate_forces

    def __len__(self) -> int:
        """Return number of registered force components."""
        return len(self._forces)

    def __contains__(self, force_instance: AbstractForce) -> bool:
        """Check if force instance is registered."""
        return force_instance in self._forces


class InputRegistry:
    """Manages input processing components independently from beam dynamics."""

    def __init__(self):
        """Initialize empty input registry."""
        self._input_handlers = []

    def register(self, input_handler: AbstractInputHandler) -> None:
        """
        Register an input handler if it's enabled.

        Args:
            input_handler: Instance implementing AbstractInputHandler
        """
        if input_handler.is_enabled():
            self._input_handlers.append(input_handler)

    def unregister(self, input_handler: AbstractInputHandler) -> bool:
        """
        Unregister an input handler.

        Args:
            input_handler: Handler instance to remove

        Returns:
            True if handler was found and removed, False otherwise
        """
        if input_handler in self._input_handlers:
            self._input_handlers.remove(input_handler)
            return True
        return False

    def clear(self) -> None:
        """Remove all registered input handlers."""
        self._input_handlers.clear()

    def get_registered_handlers(self) -> List[AbstractInputHandler]:
        """
        Get list of registered input handlers.

        Returns:
            List of registered handler instances
        """
        return self._input_handlers.copy()

    def create_aggregated_function(self) -> Callable:
        """
        Create function that aggregates input modifications from all registered handlers.

        Returns:
            Function that processes input: process_input(x, u, t) -> np.ndarray
        """

        def aggregate_input_processing(
            x: np.ndarray, u: np.ndarray, t: float = 0.0
        ) -> np.ndarray:
            """Compute combined input modifications from all registered handlers."""
            if not self._input_handlers:
                return u.copy()

            # Start with original input
            total_input = u.copy()

            # Add contributions from each handler
            for handler in self._input_handlers:
                if handler.is_enabled():
                    # Each handler contributes a modification to the input
                    input_contrib = handler.process_input(x, u, t)
                    # Add the handler's contribution (assuming handlers return modifications, not full inputs)
                    total_input += input_contrib

            return total_input

        return aggregate_input_processing

    def __len__(self) -> int:
        """Return number of registered input handlers."""
        return len(self._input_handlers)

    def __contains__(self, input_handler: AbstractInputHandler) -> bool:
        """Check if input handler is registered."""
        return input_handler in self._input_handlers
