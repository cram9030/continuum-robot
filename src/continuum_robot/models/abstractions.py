from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from typing import Optional, Dict, Callable, List, Union
from dataclasses import dataclass


class ElementType(Enum):
    """Enumeration of supported element types."""

    LINEAR = "linear"
    NONLINEAR = "nonlinear"


@dataclass
class Properties:
    """Standardized segment properties container with validation."""

    length: float
    elastic_modulus: float
    moment_inertia: float
    density: float
    cross_area: float
    segment_id: int
    element_type: str  # "linear" or "nonlinear" from CSV 'type' column

    # Optional fluid dynamics properties
    wetted_area: Optional[float] = None
    drag_coef: Optional[float] = None

    def __post_init__(self):
        """Validate properties after initialization."""
        if self.length <= 0:
            raise ValueError(f"Length must be positive, got {self.length}")
        if self.elastic_modulus <= 0:
            raise ValueError(
                f"Elastic modulus must be positive, got {self.elastic_modulus}"
            )
        if self.moment_inertia <= 0:
            raise ValueError(
                f"Moment of inertia must be positive, got {self.moment_inertia}"
            )
        if self.density <= 0:
            raise ValueError(f"Density must be positive, got {self.density}")
        if self.cross_area <= 0:
            raise ValueError(f"Cross area must be positive, got {self.cross_area}")

        # Validate element type
        valid_types = {t.value for t in ElementType}
        if self.element_type.lower() not in valid_types:
            raise ValueError(f"Invalid element type: {self.element_type}")

    def get_element_type(self) -> ElementType:
        """Get ElementType enum from string."""
        return ElementType(self.element_type.lower())

    def has_fluid_properties(self) -> bool:
        """Check if segment has fluid dynamics properties."""
        return self.wetted_area is not None and self.drag_coef is not None


@dataclass
class AssemblyContext:
    """Context information for segment assembly."""

    global_dof_offset: int  # Starting DOF index for this segment
    node_start: int  # Starting node index for this segment
    node_end: int  # Ending node index for this segment


class ISegment(ABC):
    """Unified segment interface - ALL segments use 3-DOF per node [u, w, θ]"""

    def __init__(self, properties: Properties):
        self.properties = properties
        self.segment_id = properties.segment_id

    @abstractmethod
    def get_mass_matrix(self) -> np.ndarray:
        """Return 6x6 local mass matrix [u1, w1, θ1, u2, w2, θ2]"""
        pass

    @abstractmethod
    def get_stiffness_func(
        self,
    ) -> Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        """Return 6x6 stiffness matrix (linear) or stiffness function (nonlinear)"""
        pass

    @abstractmethod
    def get_element_type(self) -> ElementType:
        """Return the element type (linear or nonlinear)"""
        pass

    def validate_properties(self) -> None:
        """Validate segment properties (already done in Properties.__post_init__)"""
        pass

    def get_properties(self) -> Properties:
        """Get segment properties"""
        return self.properties


class ISegmentFactory(ABC):
    """Factory interface for creating segments"""

    @abstractmethod
    def create_segment(self, properties: Properties) -> ISegment:
        """Create appropriate segment type from properties"""
        pass

    @abstractmethod
    def detect_element_type(self, properties: Properties) -> ElementType:
        """Detect element type from properties"""
        pass


class IBeam(ABC):
    """Unified beam interface"""

    def __init__(self, segments: List[ISegment]):
        self.segments = segments

    @abstractmethod
    def assemble_mass_matrix(self) -> np.ndarray:
        """Assemble global mass matrix from segments"""
        pass

    @abstractmethod
    def create_stiffness_function(self) -> Callable:
        """Create global stiffness function from segments"""
        pass

    @abstractmethod
    def apply_boundary_conditions(self, boundary_conditions: Dict) -> None:
        """Apply boundary conditions to the beam"""
        pass

    @abstractmethod
    def get_constrained_dofs(self) -> List[int]:
        """Get list of constrained DOF indices"""
        pass
