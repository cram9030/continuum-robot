from typing import List, Callable
from functools import wraps
import numpy as np
from .abstractions import AbstractForce, AbstractInputHandler


def force_component(*force_classes: AbstractForce):
    """
    Decorator to add force components to create_system_func.

    Args:
        *force_classes: Force component classes to add to the system

    Usage:
        @force_component(FluidDragForce)
        def create_system_func(self):
            # Implementation
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Store force components on the instance
            if not hasattr(self, "_force_components"):
                self._force_components = []

            # Initialize force components with the beam instance
            for force_class in force_classes:
                if hasattr(force_class, "__call__"):  # Check if it's a class
                    force_instance = force_class(self)
                    if force_instance.is_enabled():
                        self._force_components.append(force_instance)

            # Call the original function
            result = func(self, *args, **kwargs)
            return result

        return wrapper

    return decorator


def input_component(*input_classes: AbstractInputHandler):
    """
    Decorator to add input processing components to create_input_func.

    Args:
        *input_classes: Input handler classes to add to the system

    Usage:
        @input_component(CustomInputHandler)
        def create_input_func(self):
            # Implementation
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Store input components on the instance
            if not hasattr(self, "_input_components"):
                self._input_components = []

            # Initialize input components with the beam instance
            for input_class in input_classes:
                if hasattr(input_class, "__call__"):  # Check if it's a class
                    input_instance = input_class(self)
                    if input_instance.is_enabled():
                        self._input_components.append(input_instance)

            # Call the original function
            result = func(self, *args, **kwargs)
            return result

        return wrapper

    return decorator


def create_forces_function(force_components: List[AbstractForce]) -> Callable:
    """
    Create a forces function that combines all registered force components.

    Args:
        force_components: List of force component instances

    Returns:
        Function that computes total forces: forces(x, t) -> np.ndarray
    """

    def forces(x: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Combined forces function from all registered components."""
        if not force_components:
            # Return zero forces if no components
            n_states = len(x) // 2
            return np.zeros(n_states)

        total_forces = None
        for force_comp in force_components:
            if force_comp.is_enabled():
                comp_forces = force_comp.compute_forces(x, t)
                if total_forces is None:
                    total_forces = comp_forces.copy()
                else:
                    total_forces += comp_forces

        # Return zero forces if no enabled components
        if total_forces is None:
            n_states = len(x) // 2
            return np.zeros(n_states)

        return total_forces

    return forces


def create_input_processor(input_components: List[AbstractInputHandler]) -> Callable:
    """
    Create an input processor that combines all registered input components.

    Args:
        input_components: List of input handler instances

    Returns:
        Function that processes inputs: process_input(x, u, t) -> np.ndarray
    """

    def process_input(x: np.ndarray, u: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Combined input processor from all registered components."""
        processed_input = u.copy()

        for input_comp in input_components:
            if input_comp.is_enabled():
                processed_input = input_comp.process_input(x, processed_input, t)

        return processed_input

    return process_input
