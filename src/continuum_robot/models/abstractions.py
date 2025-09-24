from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import pandas as pd
from typing import Optional, Dict, Callable, List, Union
from dataclasses import dataclass


class ElementType(Enum):
    """Enumeration of supported element types."""

    LINEAR = "linear"
    NONLINEAR = "nonlinear"


class BoundaryConditionType(Enum):
    """Enumeration of supported boundary condition types."""

    FIXED = "fixed"  # Both displacement and rotation fixed
    PINNED = "pinned"  # Displacement fixed, rotation free


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


class AbstractForce(ABC):
    """Abstract base class for force components in dynamic systems."""

    @abstractmethod
    def compute_forces(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute force vector given current state and time.

        Args:
            x: Current state vector [positions, velocities]
            t: Current time

        Returns:
            Force vector corresponding to position states
        """
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """Return True if this force component is enabled."""
        pass


class AbstractInputHandler(ABC):
    """Abstract base class for input processing components."""

    @abstractmethod
    def compute_input(self, x: np.ndarray, r: np.ndarray, t: float) -> np.ndarray:
        """
        Compute input modifications given current state, input, and time.

        Args:
            x: Current state vector [positions, velocities]
            r: Refrence input vector
            t: Current time

        Returns:
            Input modification vector (delta) to be added to original input
        """
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """Return True if this input handler is enabled."""
        pass


def create_properties_from_dataframe(df: pd.DataFrame, segment_id: int) -> Properties:
    """
    Create Properties object from DataFrame row.

    Args:
        df: DataFrame containing beam parameters
        segment_id: Index of the segment (row) to create properties for

    Returns:
        Properties object for the segment
    """
    if segment_id >= len(df):
        raise IndexError(f"Segment ID {segment_id} exceeds DataFrame length {len(df)}")

    row = df.iloc[segment_id]

    # Required properties
    props_dict = {
        "length": row["length"],
        "elastic_modulus": row["elastic_modulus"],
        "moment_inertia": row["moment_inertia"],
        "density": row["density"],
        "cross_area": row["cross_area"],
        "segment_id": segment_id,
        "element_type": row["type"],  # This is the key field from CSV
    }

    # Optional fluid properties
    if "wetted_area" in df.columns:
        props_dict["wetted_area"] = row["wetted_area"]
    if "drag_coef" in df.columns:
        props_dict["drag_coef"] = row["drag_coef"]

    return Properties(**props_dict)
