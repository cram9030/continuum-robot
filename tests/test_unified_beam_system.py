import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from continuum_robot.models.abstractions import (
    Properties,
    ElementType,
    BoundaryConditionType,
)
from continuum_robot.models.segments import (
    LinearSegment,
    NonlinearSegment,
    SegmentFactory,
    create_properties_from_dataframe,
)
from continuum_robot.models.euler_bernoulli_beam import EulerBernoulliBeam
from continuum_robot.models.dynamic_beam_model import DynamicEulerBernoulliBeam


class TestProperties:
    """Test Properties dataclass and validation."""

    def test_valid_properties(self):
        """Test creation of valid properties."""
        props = Properties(
            length=1.0,
            elastic_modulus=200e9,
            moment_inertia=1e-6,
            density=7850,
            cross_area=1e-4,
            segment_id=0,
            element_type="linear",
        )
        assert props.length == 1.0
        assert props.get_element_type() == ElementType.LINEAR

    def test_invalid_properties_negative_length(self):
        """Test validation catches negative length."""
        with pytest.raises(ValueError, match="Length must be positive"):
            Properties(
                length=-1.0,
                elastic_modulus=200e9,
                moment_inertia=1e-6,
                density=7850,
                cross_area=1e-4,
                segment_id=0,
                element_type="linear",
            )

    def test_invalid_element_type(self):
        """Test validation catches invalid element type."""
        with pytest.raises(ValueError, match="Invalid element type"):
            Properties(
                length=1.0,
                elastic_modulus=200e9,
                moment_inertia=1e-6,
                density=7850,
                cross_area=1e-4,
                segment_id=0,
                element_type="invalid",
            )

    def test_has_fluid_properties(self):
        """Test fluid properties detection."""
        props = Properties(
            length=1.0,
            elastic_modulus=200e9,
            moment_inertia=1e-6,
            density=7850,
            cross_area=1e-4,
            segment_id=0,
            element_type="linear",
            wetted_area=0.1,
            drag_coef=1.2,
        )
        assert props.has_fluid_properties() is True


class TestLinearSegment:
    """Test LinearSegment implementation."""

    @pytest.fixture
    def linear_properties(self):
        """Create test properties for linear segment."""
        return Properties(
            length=1.0,
            elastic_modulus=200e9,
            moment_inertia=1e-6,
            density=7850,
            cross_area=1e-4,
            segment_id=0,
            element_type="linear",
        )

    def test_creation(self, linear_properties):
        """Test linear segment creation."""
        segment = LinearSegment(linear_properties)
        assert segment.get_element_type() == ElementType.LINEAR
        assert segment.segment_id == 0

    def test_mass_matrix(self, linear_properties):
        """Test mass matrix generation."""
        segment = LinearSegment(linear_properties)
        M = segment.get_mass_matrix()
        assert M.shape == (6, 6)
        assert np.allclose(M, M.T)  # Should be symmetric
        assert np.all(np.linalg.eigvals(M) > 0)  # Should be positive definite

    def test_stiffness_matrix(self, linear_properties):
        """Test stiffness matrix generation."""
        segment = LinearSegment(linear_properties)
        K = segment.get_stiffness_func()
        assert isinstance(K, np.ndarray)  # Should return matrix for linear
        assert K.shape == (6, 6)

    def test_wrong_element_type(self):
        """Test that LinearSegment rejects nonlinear element type."""
        props = Properties(
            length=1.0,
            elastic_modulus=200e9,
            moment_inertia=1e-6,
            density=7850,
            cross_area=1e-4,
            segment_id=0,
            element_type="nonlinear",
        )
        with pytest.raises(
            ValueError, match="LinearSegment requires LINEAR element type"
        ):
            LinearSegment(props)


class TestNonlinearSegment:
    """Test NonlinearSegment implementation."""

    @pytest.fixture
    def nonlinear_properties(self):
        """Create test properties for nonlinear segment."""
        return Properties(
            length=1.0,
            elastic_modulus=200e9,
            moment_inertia=1e-6,
            density=7850,
            cross_area=1e-4,
            segment_id=0,
            element_type="nonlinear",
        )

    def test_creation(self, nonlinear_properties):
        """Test nonlinear segment creation."""
        segment = NonlinearSegment(nonlinear_properties)
        assert segment.get_element_type() == ElementType.NONLINEAR
        assert segment.segment_id == 0

    def test_mass_matrix(self, nonlinear_properties):
        """Test mass matrix generation."""
        segment = NonlinearSegment(nonlinear_properties)
        M = segment.get_mass_matrix()
        assert M.shape == (6, 6)
        assert np.allclose(M, M.T)  # Should be symmetric
        assert np.all(np.linalg.eigvals(M) > 0)  # Should be positive definite

    def test_stiffness_function(self, nonlinear_properties):
        """Test stiffness function generation."""
        segment = NonlinearSegment(nonlinear_properties)
        stiffness_func = segment.get_stiffness_func()
        assert callable(stiffness_func)  # Should return function for nonlinear

        # Test with sample state
        state = np.array([0.01, 0.001, 0.1, 0.02, 0.002, 0.2])
        forces = stiffness_func(state)
        assert forces.shape == (6,)
        assert np.isfinite(forces).all()


class TestSegmentFactory:
    """Test SegmentFactory functionality."""

    def test_create_linear_segment(self):
        """Test factory creates linear segment."""
        props = Properties(
            length=1.0,
            elastic_modulus=200e9,
            moment_inertia=1e-6,
            density=7850,
            cross_area=1e-4,
            segment_id=0,
            element_type="linear",
        )
        factory = SegmentFactory()
        segment = factory.create_segment(props)
        assert isinstance(segment, LinearSegment)
        assert segment.get_element_type() == ElementType.LINEAR

    def test_create_nonlinear_segment(self):
        """Test factory creates nonlinear segment."""
        props = Properties(
            length=1.0,
            elastic_modulus=200e9,
            moment_inertia=1e-6,
            density=7850,
            cross_area=1e-4,
            segment_id=0,
            element_type="nonlinear",
        )
        factory = SegmentFactory()
        segment = factory.create_segment(props)
        assert isinstance(segment, NonlinearSegment)
        assert segment.get_element_type() == ElementType.NONLINEAR

    def test_detect_element_type(self):
        """Test element type detection."""
        factory = SegmentFactory()

        linear_props = Properties(
            length=1.0,
            elastic_modulus=200e9,
            moment_inertia=1e-6,
            density=7850,
            cross_area=1e-4,
            segment_id=0,
            element_type="linear",
        )
        assert factory.detect_element_type(linear_props) == ElementType.LINEAR

        nonlinear_props = Properties(
            length=1.0,
            elastic_modulus=200e9,
            moment_inertia=1e-6,
            density=7850,
            cross_area=1e-4,
            segment_id=0,
            element_type="nonlinear",
        )
        assert factory.detect_element_type(nonlinear_props) == ElementType.NONLINEAR


class TestCreatePropertiesFromDataFrame:
    """Test create_properties_from_dataframe function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "length": [1.0, 2.0],
                "elastic_modulus": [200e9, 150e9],
                "moment_inertia": [1e-6, 2e-6],
                "density": [7850, 8000],
                "cross_area": [1e-4, 2e-4],
                "type": ["linear", "nonlinear"],
                "wetted_area": [0.1, 0.2],
                "drag_coef": [1.2, 1.5],
            }
        )

    def test_create_properties(self, sample_dataframe):
        """Test properties creation from DataFrame."""
        props = create_properties_from_dataframe(sample_dataframe, 0)
        assert props.length == 1.0
        assert props.element_type == "linear"
        assert props.segment_id == 0
        assert props.wetted_area == 0.1
        assert props.has_fluid_properties() is True

    def test_invalid_segment_id(self, sample_dataframe):
        """Test error handling for invalid segment ID."""
        with pytest.raises(IndexError, match="Segment ID 5 exceeds DataFrame length"):
            create_properties_from_dataframe(sample_dataframe, 5)


class TestEulerBernoulliBeam:
    """Test unified EulerBernoulliBeam implementation."""

    @pytest.fixture
    def linear_beam_df(self):
        """Create DataFrame for pure linear beam."""
        return pd.DataFrame(
            {
                "length": [1.0, 1.0],
                "elastic_modulus": [200e9, 200e9],
                "moment_inertia": [1e-6, 1e-6],
                "density": [7850, 7850],
                "cross_area": [1e-4, 1e-4],
                "type": ["linear", "linear"],
            }
        )

    @pytest.fixture
    def nonlinear_beam_df(self):
        """Create DataFrame for pure nonlinear beam."""
        return pd.DataFrame(
            {
                "length": [1.0, 1.0],
                "elastic_modulus": [200e9, 200e9],
                "moment_inertia": [1e-6, 1e-6],
                "density": [7850, 7850],
                "cross_area": [1e-4, 1e-4],
                "type": ["nonlinear", "nonlinear"],
            }
        )

    @pytest.fixture
    def hybrid_beam_df(self):
        """Create DataFrame for hybrid beam."""
        return pd.DataFrame(
            {
                "length": [1.0, 1.0],
                "elastic_modulus": [200e9, 200e9],
                "moment_inertia": [1e-6, 1e-6],
                "density": [7850, 7850],
                "cross_area": [1e-4, 1e-4],
                "type": ["linear", "nonlinear"],
            }
        )

    def test_pure_linear_beam(self, linear_beam_df):
        """Test creation of pure linear beam."""
        beam = EulerBernoulliBeam(linear_beam_df)
        assert beam.get_segment_count() == 2
        assert beam.is_hybrid() is False
        assert all(t == ElementType.LINEAR for t in beam.get_segment_types())

    def test_pure_nonlinear_beam(self, nonlinear_beam_df):
        """Test creation of pure nonlinear beam."""
        beam = EulerBernoulliBeam(nonlinear_beam_df)
        assert beam.get_segment_count() == 2
        assert beam.is_hybrid() is False
        assert all(t == ElementType.NONLINEAR for t in beam.get_segment_types())

    def test_hybrid_beam(self, hybrid_beam_df):
        """Test creation of hybrid beam."""
        beam = EulerBernoulliBeam(hybrid_beam_df)
        assert beam.get_segment_count() == 2
        assert beam.is_hybrid() is True
        types = beam.get_segment_types()
        assert ElementType.LINEAR in types
        assert ElementType.NONLINEAR in types

    def test_mass_matrix_assembly(self, hybrid_beam_df):
        """Test mass matrix assembly for hybrid beam."""
        beam = EulerBernoulliBeam(hybrid_beam_df)
        M = beam.get_mass_matrix()
        expected_size = 3 * (2 + 1)  # 3 DOFs per node, 2 segments + 1 = 3 nodes
        assert M.shape == (expected_size, expected_size)
        assert np.allclose(M, M.T)  # Should be symmetric
        assert np.all(np.linalg.eigvals(M) > 0)  # Should be positive definite

    def test_stiffness_function(self, hybrid_beam_df):
        """Test stiffness function for hybrid beam."""
        beam = EulerBernoulliBeam(hybrid_beam_df)
        stiffness_func = beam.get_stiffness_function()
        assert callable(stiffness_func)

        # Test with sample state
        n_dofs = beam.get_mass_matrix().shape[0]
        state = np.random.random(n_dofs) * 0.01  # Small displacements
        forces = stiffness_func(state)
        assert forces.shape == (n_dofs,)
        assert np.isfinite(forces).all()

    def test_boundary_conditions(self, hybrid_beam_df):
        """Test boundary condition application."""
        beam = EulerBernoulliBeam(hybrid_beam_df)
        original_size = beam.get_mass_matrix().shape[0]

        # Apply fixed boundary condition at first node
        conditions = {0: BoundaryConditionType.FIXED}
        beam.apply_boundary_conditions(conditions)

        # Should reduce DOFs by 3 (u, w, phi at node 0)
        new_size = beam.get_mass_matrix().shape[0]
        assert new_size == original_size - 3

        # Check that boundary conditions are tracked
        assert beam.has_boundary_conditions()
        assert beam.get_boundary_conditions() == conditions
        assert len(beam.get_constrained_dofs()) == 3

    def test_dof_mapping(self, hybrid_beam_df):
        """Test DOF mapping functionality."""
        beam = EulerBernoulliBeam(hybrid_beam_df)

        # Test initial mapping
        assert beam.get_dof_to_node_param(0) == ("u", 0)
        assert beam.get_dof_to_node_param(1) == ("w", 0)
        assert beam.get_dof_to_node_param(2) == ("phi", 0)

        assert beam.get_dof_index(1, "w") == 4
        assert beam.get_dof_index(2, "phi") == 8

    def test_clear_boundary_conditions(self, hybrid_beam_df):
        """Test clearing boundary conditions."""
        beam = EulerBernoulliBeam(hybrid_beam_df)
        original_size = beam.get_mass_matrix().shape[0]

        # Apply boundary conditions
        conditions = {0: BoundaryConditionType.FIXED}
        beam.apply_boundary_conditions(conditions)

        # Clear boundary conditions
        beam.clear_boundary_conditions()

        # Should restore original size
        assert beam.get_mass_matrix().shape[0] == original_size
        assert not beam.has_boundary_conditions()
        assert len(beam.get_boundary_conditions()) == 0


class TestDynamicBeamModelWithUnified:
    """Test DynamicEulerBernoulliBeam with unified system."""

    @pytest.fixture
    def mixed_beam_csv(self):
        """Create temporary CSV file with mixed element types."""
        df = pd.DataFrame(
            {
                "length": [1.0, 1.0],
                "elastic_modulus": [200e9, 200e9],
                "moment_inertia": [1e-6, 1e-6],
                "density": [7850, 7850],
                "cross_area": [1e-4, 1e-4],
                "type": ["linear", "nonlinear"],
                "boundary_condition": ["NONE", "NONE"],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            yield f.name

        os.unlink(f.name)

    def test_mixed_system_creation(self, mixed_beam_csv):
        """Test that mixed systems now work with unified beam."""
        # This should no longer raise an error
        dynamic_beam = DynamicEulerBernoulliBeam(mixed_beam_csv)

        # Should have unified model, not separate models
        assert hasattr(dynamic_beam, "beam_model")
        assert dynamic_beam.beam_model is not None

    def test_mixed_system_functions(self, mixed_beam_csv):
        """Test system function creation for mixed system."""
        dynamic_beam = DynamicEulerBernoulliBeam(mixed_beam_csv)

        # Create system functions
        dynamic_beam.create_system_func()
        dynamic_beam.create_input_func()

        # Should have created system function
        system_func = dynamic_beam.get_system_func()
        assert callable(system_func)

        # Test with sample state
        n_dofs = dynamic_beam.beam_model.get_mass_matrix().shape[0]
        state = np.random.random(2 * n_dofs) * 0.01  # [positions, velocities]
        result = system_func(state)
        assert result.shape == state.shape
        assert np.isfinite(result).all()

    def test_mixed_system_dynamic_integration(self, mixed_beam_csv):
        """Test that mixed system can be used with solve_ivp."""
        dynamic_beam = DynamicEulerBernoulliBeam(mixed_beam_csv)
        dynamic_beam.create_system_func()
        dynamic_beam.create_input_func()

        # Get dynamic system
        dynamic_system = dynamic_beam.get_dynamic_system()

        # Test dynamic system function
        n_dofs = dynamic_beam.beam_model.get_mass_matrix().shape[0]
        state = np.random.random(2 * n_dofs) * 0.01
        force = np.zeros(n_dofs)

        result = dynamic_system(0.0, state, force)
        assert result.shape == state.shape
        assert np.isfinite(result).all()


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete workflow scenarios."""

    def test_complete_hybrid_workflow(self):
        """Test complete workflow with hybrid beam."""
        # Create hybrid beam data
        df = pd.DataFrame(
            {
                "length": [1.0, 1.0, 1.0],
                "elastic_modulus": [200e9, 200e9, 200e9],
                "moment_inertia": [1e-6, 1e-6, 1e-6],
                "density": [7850, 7850, 7850],
                "cross_area": [1e-4, 1e-4, 1e-4],
                "type": ["linear", "nonlinear", "linear"],
                "boundary_condition": ["FIXED", "NONE", "NONE"],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)

            try:
                # Create dynamic beam
                dynamic_beam = DynamicEulerBernoulliBeam(f.name)

                # Verify it's using unified model
                assert dynamic_beam.beam_model is not None
                assert dynamic_beam.beam_model.is_hybrid()

                # Create system functions
                dynamic_beam.create_system_func()
                dynamic_beam.create_input_func()

                # Get system for integration
                system = dynamic_beam.get_dynamic_system()

                # Test with realistic initial conditions
                n_dofs = dynamic_beam.beam_model.get_mass_matrix().shape[0]
                initial_state = np.zeros(2 * n_dofs)
                initial_state[n_dofs:] = 0.01  # Small initial velocity

                # Test system evaluation
                result = system(0.0, initial_state, np.zeros(n_dofs))

                # Verify results are reasonable
                assert result.shape == initial_state.shape
                assert np.isfinite(result).all()

            finally:
                os.unlink(f.name)
