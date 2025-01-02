import pytest
import numpy as np
import pandas as pd
import pathlib
from scipy import sparse
import tempfile
import os

from linear_euler_bernoulli_beam import LinearEulerBernoulliBeam

@pytest.fixture
def nitinol_file():
    """Create a temporary CSV file with Nitinol beam parameters."""
    # Create temp file with example Nitinol parameters
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        # Header
        f.write("length,elastic_modulus,moment_inertia,density,cross_area\n")
        
        # Example Nitinol parameters (SI units)
        # E = 75e9 Pa
        # ρ = 6450 kg/m³
        # r = 0.005 m
        # I = πr⁴/4 ≈ 4.91e-10 m⁴
        # A = πr² ≈ 7.85e-5 m²
        
        # Split 1m beam into 4 sections
        for _ in range(4):
            f.write(f"0.25,75e9,4.91e-10,6450,7.85e-5\n")
    
    yield f.name
    # Cleanup temp file
    os.unlink(f.name)

@pytest.fixture
def invalid_file():
    """Create a temporary CSV file with invalid parameters."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("length,elastic_modulus,moment_inertia,density,cross_area\n")
        f.write("0.25,75e9,-4.91e-10,6450,7.85e-5\n")  # Negative moment of inertia
    
    yield f.name
    os.unlink(f.name)

def test_initialization(nitinol_file):
    """Test basic initialization with valid file."""
    beam = LinearEulerBernoulliBeam(nitinol_file)
    assert beam is not None
    assert len(beam.parameters) == 4
    assert beam.get_length() == pytest.approx(1.0)

def test_invalid_file():
    """Test initialization with nonexistent file."""
    with pytest.raises(FileNotFoundError):
        LinearEulerBernoulliBeam("nonexistent.csv")

def test_invalid_parameters(invalid_file):
    """Test initialization with invalid parameters."""
    with pytest.raises(ValueError):
        LinearEulerBernoulliBeam(invalid_file)

def test_stiffness_matrix(nitinol_file):
    """Test stiffness matrix creation and retrieval."""
    beam = LinearEulerBernoulliBeam(nitinol_file)
    
    # Test matrix creation
    beam.create_stiffness_matrix()
    K = beam.get_stiffness_matrix()
    
    # Check matrix properties
    assert isinstance(K, np.ndarray)
    assert K.shape == (10, 10)  # 4 segments = 10x10 matrix
    assert np.allclose(K, K.T)  # Should be symmetric
    
    # Test single segment retrieval
    K_segment = beam.get_segment_stiffness(0)
    assert K_segment.shape == (4, 4)
    assert np.allclose(K_segment, K_segment.T)  # Should be symmetric

def test_mass_matrix(nitinol_file):
    """Test mass matrix creation and retrieval."""
    beam = LinearEulerBernoulliBeam(nitinol_file)
    
    # Test matrix creation
    beam.create_mass_matrix()
    M = beam.get_mass_matrix()
    
    # Check matrix properties
    assert isinstance(M, np.ndarray)
    assert M.shape == (10, 10)  # 4 segments = 10x10 matrix
    assert np.allclose(M, M.T)  # Should be symmetric
    
    # Test single segment retrieval
    M_segment = beam.get_segment_mass(0)
    assert M_segment.shape == (4, 4)
    assert np.allclose(M_segment, M_segment.T)  # Should be symmetric

def test_matrix_access_before_creation(nitinol_file):
    """Test accessing matrices before they're created."""
    beam = LinearEulerBernoulliBeam(nitinol_file)
    
    with pytest.raises(RuntimeError):
        beam.get_stiffness_matrix()
    
    with pytest.raises(RuntimeError):
        beam.get_mass_matrix()

def test_segment_index_bounds(nitinol_file):
    """Test segment access with invalid indices."""
    beam = LinearEulerBernoulliBeam(nitinol_file)
    
    with pytest.raises(IndexError):
        beam.get_segment_stiffness(-1)
    
    with pytest.raises(IndexError):
        beam.get_segment_stiffness(4)
    
    with pytest.raises(IndexError):
        beam.get_segment_mass(-1)
    
    with pytest.raises(IndexError):
        beam.get_segment_mass(4)

def test_parameter_update(nitinol_file):
    """Test matrix updates after reading new parameters."""
    beam = LinearEulerBernoulliBeam(nitinol_file)
    beam.create_stiffness_matrix()
    beam.create_mass_matrix()
    
    # Read same file again
    beam.read_parameter_file(nitinol_file)
    
    # Matrices should be reset
    with pytest.raises(RuntimeError):
        beam.get_stiffness_matrix()
    
    with pytest.raises(RuntimeError):
        beam.get_mass_matrix()

def test_matrix_values(nitinol_file):
    """Test specific matrix values for a single segment."""
    beam = LinearEulerBernoulliBeam(nitinol_file)
    
    # Get first segment matrices
    K = beam.get_segment_stiffness(0)
    M = beam.get_segment_mass(0)
    
    # Test key properties of matrices
    # K should be positive definite
    eigenvals = np.linalg.eigvals(K)
    assert np.all(eigenvals > 0)
    
    # M should be positive definite
    eigenvals = np.linalg.eigvals(M)
    assert np.all(eigenvals > 0)