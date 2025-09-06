import numpy as np
import pandas as pd
from typing import Union, Callable
from functools import partial

from .abstractions import ISegment, ISegmentFactory, Properties, ElementType


class LinearSegment(ISegment):
    """Linear segment implementation using exact functions from LinearEulerBernoulliBeam."""

    def __init__(self, properties: Properties):
        super().__init__(properties)
        if properties.get_element_type() != ElementType.LINEAR:
            raise ValueError(
                f"LinearSegment requires LINEAR element type, got {properties.element_type}"
            )

    def get_mass_matrix(self) -> np.ndarray:
        """Return 6x6 local mass matrix using exact function from LinearEulerBernoulliBeam."""
        return self._calculate_segment_mass()

    def get_stiffness_func(
        self,
    ) -> Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        """Return constant 6x6 stiffness matrix using exact function from LinearEulerBernoulliBeam."""
        return self._calculate_segment_stiffness()

    def get_element_type(self) -> ElementType:
        """Return LINEAR element type."""
        return ElementType.LINEAR

    def _calculate_segment_stiffness(self) -> np.ndarray:
        """Calculate local stiffness matrix - exact copy from LinearEulerBernoulliBeam."""
        L = self.properties.length
        EI = self.properties.elastic_modulus * self.properties.moment_inertia
        EA = self.properties.elastic_modulus * self.properties.cross_area

        # 6x6 stiffness matrix with DOFs [u1, w1, phi1, u2, w2, phi2]
        return np.array(
            [
                [EA / L, 0, 0, -EA / L, 0, 0],
                [
                    0,
                    12 * EI / L**3,
                    6 * EI / L**2,
                    0,
                    -12 * EI / L**3,
                    6 * EI / L**2,
                ],
                [0, 6 * EI / L**2, 4 * EI / L, 0, -6 * EI / L**2, 2 * EI / L],
                [-EA / L, 0, 0, EA / L, 0, 0],
                [
                    0,
                    -12 * EI / L**3,
                    -6 * EI / L**2,
                    0,
                    12 * EI / L**3,
                    -6 * EI / L**2,
                ],
                [0, 6 * EI / L**2, 2 * EI / L, 0, -6 * EI / L**2, 4 * EI / L],
            ]
        )

    def _calculate_segment_mass(self) -> np.ndarray:
        """Calculate local mass matrix - exact copy from LinearEulerBernoulliBeam."""
        L = self.properties.length
        rhoA = self.properties.density * self.properties.cross_area

        return np.array(
            [
                [140, 0, 0, 70, 0, 0],
                [0, 156, -22 * L, 0, 54, 13 * L],
                [0, -22 * L, 4 * L**2, 0, -13 * L, -3 * L**2],
                [70, 0, 0, 140, 0, 0],
                [0, 54, -13 * L, 0, 156, 22 * L],
                [0, 13 * L, -3 * L**2, 0, 22 * L, 4 * L**2],
            ]
        ) * (rhoA * L / 420)


class NonlinearSegment(ISegment):
    """Nonlinear segment implementation using exact functions from NonlinearEulerBernoulliBeam."""

    def __init__(self, properties: Properties):
        super().__init__(properties)
        if properties.get_element_type() != ElementType.NONLINEAR:
            raise ValueError(
                f"NonlinearSegment requires NONLINEAR element type, got {properties.element_type}"
            )

    def get_mass_matrix(self) -> np.ndarray:
        """Return 6x6 local mass matrix using exact function from NonlinearEulerBernoulliBeam."""
        return self._calculate_segment_mass()

    def get_stiffness_func(
        self,
    ) -> Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        """Return state-dependent stiffness function using exact function from NonlinearEulerBernoulliBeam."""
        return self._calculate_segment_stiffness_function()

    def get_element_type(self) -> ElementType:
        """Return NONLINEAR element type."""
        return ElementType.NONLINEAR

    def _calculate_segment_mass(self) -> np.ndarray:
        """Calculate mass matrix - exact copy from NonlinearEulerBernoulliBeam."""
        L = self.properties.length
        rhoA = self.properties.density * self.properties.cross_area

        return np.array(
            [
                [140, 0, 0, 70, 0, 0],
                [0, 156, -22 * L, 0, 54, 13 * L],
                [0, -22 * L, 4 * L**2, 0, -13 * L, -3 * L**2],
                [70, 0, 0, 140, 0, 0],
                [0, 54, -13 * L, 0, 156, 22 * L],
                [0, 13 * L, -3 * L**2, 0, 22 * L, 4 * L**2],
            ]
        ) * (rhoA * L / 420)

    def _calculate_segment_stiffness_function(self) -> Callable:
        """
        Calculate nonlinear stiffness function - exact copy from NonlinearEulerBernoulliBeam.

        Returns:
            Callable that computes segment stiffness given state vector
        """
        seg_l = self.properties.length
        EI = self.properties.elastic_modulus * self.properties.moment_inertia
        EA = self.properties.elastic_modulus * self.properties.cross_area

        def stiffness_func(x: np.ndarray) -> np.ndarray:
            # Get states for this segment
            u1, w1, theta1, u2, w2, theta2 = x

            # Implement the nonlinear stiffness function f1-f6
            # from the provided equations
            f1 = partial(self._f_1_expr, length=seg_l, A_xx=EA)
            f2 = partial(self._f_2_expr, length=seg_l, A_xx=EA)
            f3 = partial(self._f_3_expr, length=seg_l, A_xx=EA, D_xx=EI)
            f4 = partial(self._f_4_expr, length=seg_l, A_xx=EA, D_xx=EI)
            f5 = partial(self._f_5_expr, length=seg_l, A_xx=EA, D_xx=EI)
            f6 = partial(self._f_6_expr, length=seg_l, A_xx=EA, D_xx=EI)

            # Evaluate with remaining variables
            return np.array(
                [
                    f1(u1, w1, theta1, u2, w2, theta2),
                    f3(u1, w1, theta1, u2, w2, theta2),
                    f4(u1, w1, theta1, u2, w2, theta2),
                    f2(u1, w1, theta1, u2, w2, theta2),
                    f5(u1, w1, theta1, u2, w2, theta2),
                    f6(u1, w1, theta1, u2, w2, theta2),
                ]
            )

        return stiffness_func

    def _f_1_expr(self, u1, w1, theta1, u2, w2, theta2, *, length: float, A_xx: float):
        """
        Expression for axial force at node 1 - exact copy from NonlinearEulerBernoulliBeam.

        Parameters:
        u1 (float): Displacement at node 1.
        w1 (float): Transverse displacement at node 1.
        theta1 (float): Rotation at node 1.
        u2 (float): Displacement at node 2.
        w2 (float): Transverse displacement at node 2.
        theta2 (float): Rotation at node 2.
        length (float): Length of the beam element.
        A_xx (float): Elastic modulus times cross-sectional area (EA).

        Returns:
        float: Axial force at node 1.
        """
        # A_xx is elastic modulus * cross sectional area (EA)
        # D_xx is elastic modulus * moment of inertia (EI)
        return (
            A_xx
            * (
                length
                * (
                    -theta1
                    * (
                        0.0666666666666665 * theta1 * length
                        - 0.0166666666666667 * theta2 * length
                        - 0.05 * w1
                        + 0.05 * w2
                    )
                    + theta2
                    * (
                        0.0166666666666667 * theta1 * length
                        - 0.0666666666666667 * theta2 * length
                        + 0.05 * w1
                        - 0.05 * w2
                    )
                    + u1
                )
                + (-u2 - w1 + w2)
                * (
                    -0.05 * theta1 * length
                    - 0.05 * theta2 * length
                    + 0.6 * w1
                    - 0.6 * w2
                )
            )
            / length**2
        )

    def _f_2_expr(self, u1, w1, theta1, u2, w2, theta2, *, length: float, A_xx: float):
        """
        Expression for bending force at node 1 - exact copy from NonlinearEulerBernoulliBeam.

        Parameters:
        u1 (float): Displacement at node 1.
        w1 (float): Transverse displacement at node 1.
        theta1 (float): Rotation at node 1.
        u2 (float): Displacement at node 2.
        w2 (float): Transverse displacement at node 2.
        theta2 (float): Rotation at node 2.
        length (float): Length of the beam element.
        A_xx (float): Elastic modulus times cross-sectional area (EA).

        Returns:
        float: bending force at node 1.
        """
        return (
            A_xx
            * (
                length
                * (
                    theta1
                    * (
                        0.0666666666666665 * theta1 * length
                        - 0.0166666666666667 * theta2 * length
                        - 0.05 * w1
                        + 0.05 * w2
                    )
                    - theta2
                    * (
                        0.0166666666666667 * theta1 * length
                        - 0.0666666666666667 * theta2 * length
                        + 0.05 * w1
                        - 0.05 * w2
                    )
                    - u1
                    + u2
                )
                + (w1 - w2)
                * (
                    -0.05 * theta1 * length
                    - 0.05 * theta2 * length
                    + 0.6 * w1
                    - 0.6 * w2
                )
            )
            / length**2
        )

    def _f_3_expr(
        self, u1, w1, theta1, u2, w2, theta2, *, length: float, A_xx: float, D_xx: float
    ):
        """
        Expression for bending moment at node 1 - exact copy from NonlinearEulerBernoulliBeam.

        Parameters:
        u1 (float): Displacement at node 1.
        w1 (float): Transverse displacement at node 1.
        theta1 (float): Rotation at node 1.
        u2 (float): Displacement at node 2.
        w2 (float): Transverse displacement at node 2.
        theta2 (float): Rotation at node 2.
        length (float): Length of the beam element.
        A_xx (float): Elastic modulus times cross-sectional area (EA).

        Returns:
        float: bending moment at node 1.
        """
        return (
            0.1
            * (
                0.0357142857143344 * A_xx * theta1**3 * length**3
                - 0.107142857143003 * A_xx * theta1**2 * theta2 * length**3
                + 1.28571428571433 * A_xx * theta1**2 * length**2 * w1
                - 1.28571428571433 * A_xx * theta1**2 * length**2 * w2
                - 0.107142857143003 * A_xx * theta1 * theta2**2 * length**3
                + 1.0 * A_xx * theta1 * length**2 * u1
                - 1.0 * A_xx * theta1 * length**2 * u2
                - 3.8571428571413 * A_xx * theta1 * length * w1**2
                + 7.7142857142826 * A_xx * theta1 * length * w1 * w2
                - 3.8571428571413 * A_xx * theta1 * length * w2**2
                + 0.0357142857143344 * A_xx * theta2**3 * length**3
                + 1.28571428571433 * A_xx * theta2**2 * length**2 * w1
                - 1.28571428571433 * A_xx * theta2**2 * length**2 * w2
                + 1.0 * A_xx * theta2 * length**2 * u1
                - 1.0 * A_xx * theta2 * length**2 * u2
                - 3.857142857143 * A_xx * theta2 * length * w1**2
                + 7.71428571428601 * A_xx * theta2 * length * w1 * w2
                - 3.857142857143 * A_xx * theta2 * length * w2**2
                - 12.0 * A_xx * length * u1 * w1
                + 12.0 * A_xx * length * u1 * w2
                + 12.0 * A_xx * length * u2 * w1
                - 12.0 * A_xx * length * u2 * w2
                + 10.2857142857147 * A_xx * w1**3
                - 30.857142857144 * A_xx * w1**2 * w2
                + 30.857142857144 * A_xx * w1 * w2**2
                - 10.2857142857147 * A_xx * w2**3
                - 60.0 * D_xx * theta1 * length
                - 60.0 * D_xx * theta2 * length
                + 120.0 * D_xx * w1
                - 120.0 * D_xx * w2
            )
            / length**3
        )

    def _f_4_expr(
        self, u1, w1, theta1, u2, w2, theta2, *, length: float, A_xx: float, D_xx: float
    ):
        """
        Expression for axial force at node 2 - exact copy from NonlinearEulerBernoulliBeam.

        Parameters:
        u1 (float): Displacement at node 1.
        w1 (float): Transverse displacement at node 1.
        theta1 (float): Rotation at node 1.
        u2 (float): Displacement at node 2.
        w2 (float): Transverse displacement at node 2.
        theta2 (float): Rotation at node 2.
        length (float): Length of the beam element.
        A_xx (float): Elastic modulus times cross-sectional area (EA).

        Returns:
        float: axial force at node 2.
        """
        return (
            0.0285714285714391 * A_xx * theta1**3 * length
            - 0.0107142857142861 * A_xx * theta1**2 * theta2 * length
            + 0.0107142857142719 * A_xx * theta1**2 * w1
            - 0.0107142857142719 * A_xx * theta1**2 * w2
            + 0.00714285714286444 * A_xx * theta1 * theta2**2 * length
            - 0.0214285714286007 * A_xx * theta1 * theta2 * w1
            + 0.0214285714286007 * A_xx * theta1 * theta2 * w2
            - 0.133333333333333 * A_xx * theta1 * u1
            + 0.133333333333333 * A_xx * theta1 * u2
            + 0.128571428571433 * A_xx * theta1 * w1**2 / length
            - 0.257142857142867 * A_xx * theta1 * w1 * w2 / length
            + 0.128571428571433 * A_xx * theta1 * w2**2 / length
            - 0.00357142857143344 * A_xx * theta2**3 * length
            - 0.0107142857142719 * A_xx * theta2**2 * w1
            + 0.0107142857142719 * A_xx * theta2**2 * w2
            + 0.0333333333333333 * A_xx * theta2 * u1
            - 0.0333333333333333 * A_xx * theta2 * u2
            + 0.1 * A_xx * u1 * w1 / length
            - 0.1 * A_xx * u1 * w2 / length
            - 0.1 * A_xx * u2 * w1 / length
            + 0.1 * A_xx * u2 * w2 / length
            - 0.128571428571377 * A_xx * w1**3 / length**2
            + 0.38571428571413 * A_xx * w1**2 * w2 / length**2
            - 0.38571428571413 * A_xx * w1 * w2**2 / length**2
            + 0.128571428571377 * A_xx * w2**3 / length**2
            + 4.0 * D_xx * theta1 / length
            + 2.0 * D_xx * theta2 / length
            - 6.0 * D_xx * w1 / length**2
            + 6.0 * D_xx * w2 / length**2
        )

    def _f_5_expr(
        self, u1, w1, theta1, u2, w2, theta2, *, length: float, A_xx: float, D_xx: float
    ):
        """
        Expression for bending force at node 2 - exact copy from NonlinearEulerBernoulliBeam.

        Parameters:
        u1 (float): Displacement at node 1.
        w1 (float): Transverse displacement at node 1.
        theta1 (float): Rotation at node 1.
        u2 (float): Displacement at node 2.
        w2 (float): Transverse displacement at node 2.
        theta2 (float): Rotation at node 2.
        length (float): Length of the beam element.
        A_xx (float): Elastic modulus times cross-sectional area (EA).

        Returns:
        float: bending force at node 2.
        """
        return (
            0.1
            * (
                -0.0357142857143344 * A_xx * theta1**3 * length**3
                + 0.107142857143003 * A_xx * theta1**2 * theta2 * length**3
                - 1.28571428571433 * A_xx * theta1**2 * length**2 * w1
                + 1.28571428571433 * A_xx * theta1**2 * length**2 * w2
                + 0.107142857143003 * A_xx * theta1 * theta2**2 * length**3
                - 1.0 * A_xx * theta1 * length**2 * u1
                + 1.0 * A_xx * theta1 * length**2 * u2
                + 3.8571428571413 * A_xx * theta1 * length * w1**2
                - 7.7142857142826 * A_xx * theta1 * length * w1 * w2
                + 3.8571428571413 * A_xx * theta1 * length * w2**2
                - 0.0357142857143344 * A_xx * theta2**3 * length**3
                - 1.28571428571433 * A_xx * theta2**2 * length**2 * w1
                + 1.28571428571433 * A_xx * theta2**2 * length**2 * w2
                - 1.0 * A_xx * theta2 * length**2 * u1
                + 1.0 * A_xx * theta2 * length**2 * u2
                + 3.857142857143 * A_xx * theta2 * length * w1**2
                - 7.71428571428601 * A_xx * theta2 * length * w1 * w2
                + 3.857142857143 * A_xx * theta2 * length * w2**2
                + 12.0 * A_xx * length * u1 * w1
                - 12.0 * A_xx * length * u1 * w2
                - 12.0 * A_xx * length * u2 * w1
                + 12.0 * A_xx * length * u2 * w2
                - 10.2857142857147 * A_xx * w1**3
                + 30.857142857144 * A_xx * w1**2 * w2
                - 30.857142857144 * A_xx * w1 * w2**2
                + 10.2857142857147 * A_xx * w2**3
                + 60.0 * D_xx * theta1 * length
                + 60.0 * D_xx * theta2 * length
                - 120.0 * D_xx * w1
                + 120.0 * D_xx * w2
            )
            / length**3
        )

    def _f_6_expr(
        self, u1, w1, theta1, u2, w2, theta2, *, length: float, A_xx: float, D_xx: float
    ):
        """
        Expression for bending moment at node 2 - exact copy from NonlinearEulerBernoulliBeam.

        Parameters:
        u1 (float): Displacement at node 1.
        w1 (float): Transverse displacement at node 1.
        theta1 (float): Rotation at node 1.
        u2 (float): Displacement at node 2.
        w2 (float): Transverse displacement at node 2.
        theta2 (float): Rotation at node 2.
        length (float): Length of the beam element.
        A_xx (float): Elastic modulus times cross-sectional area (EA).

        Returns:
        float: bending moment at node 2.
        """
        return (
            -0.00357142857143344 * A_xx * theta1**3 * length
            + 0.00714285714286356 * A_xx * theta1**2 * theta2 * length
            - 0.0107142857143003 * A_xx * theta1**2 * w1
            + 0.0107142857143003 * A_xx * theta1**2 * w2
            - 0.0107142857142932 * A_xx * theta1 * theta2**2 * length
            - 0.021428571428558 * A_xx * theta1 * theta2 * w1
            + 0.021428571428558 * A_xx * theta1 * theta2 * w2
            + 0.0333333333333333 * A_xx * theta1 * u1
            - 0.0333333333333333 * A_xx * theta1 * u2
            + 0.0285714285714271 * A_xx * theta2**3 * length
            + 0.0107142857142932 * A_xx * theta2**2 * w1
            - 0.0107142857142932 * A_xx * theta2**2 * w2
            - 0.133333333333333 * A_xx * theta2 * u1
            + 0.133333333333333 * A_xx * theta2 * u2
            + 0.128571428571428 * A_xx * theta2 * w1**2 / length
            - 0.257142857142856 * A_xx * theta2 * w1 * w2 / length
            + 0.128571428571428 * A_xx * theta2 * w2**2 / length
            + 0.1 * A_xx * u1 * w1 / length
            - 0.1 * A_xx * u1 * w2 / length
            - 0.1 * A_xx * u2 * w1 / length
            + 0.1 * A_xx * u2 * w2 / length
            - 0.128571428571433 * A_xx * w1**3 / length**2
            + 0.3857142857143 * A_xx * w1**2 * w2 / length**2
            - 0.3857142857143 * A_xx * w1 * w2**2 / length**2
            + 0.128571428571433 * A_xx * w2**3 / length**2
            + 2.0 * D_xx * theta1 / length
            + 4.0 * D_xx * theta2 / length
            - 6.0 * D_xx * w1 / length**2
            + 6.0 * D_xx * w2 / length**2
        )


class SegmentFactory(ISegmentFactory):
    """Factory for creating appropriate segment types from properties."""

    def create_segment(self, properties: Properties) -> ISegment:
        """Create appropriate segment type based on element_type in properties."""
        element_type = self.detect_element_type(properties)

        if element_type == ElementType.LINEAR:
            return LinearSegment(properties)
        elif element_type == ElementType.NONLINEAR:
            return NonlinearSegment(properties)
        else:
            raise ValueError(f"Unknown element type: {element_type}")

    def detect_element_type(self, properties: Properties) -> ElementType:
        """Detect element type from properties using the 'type' column value."""
        return properties.get_element_type()


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
