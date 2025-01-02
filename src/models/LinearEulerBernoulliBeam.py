import numpy as np
import pandas as pd
from scipy import sparse
from typing import Optional, Union
import pathlib

class LinearEulerBernoulliBeam:
    """
    A class implementing the Linear Euler-Bernoulli Beam Model.
    
    This class creates and manipulates mass and stiffness matrices for a beam 
    discretized into finite elements, where each element can have different 
    material and geometric properties.
    
    The matrices are stored internally as sparse matrices but returned as dense
    matrices through getter functions.
    
    Attributes:
        parameters (pd.DataFrame): Dataframe containing beam section parameters
        K (sparse.csr_matrix): Global stiffness matrix in sparse format
        M (sparse.csr_matrix): Global mass matrix in sparse format
    """
    
    def __init__(self, filename: Union[str, pathlib.Path]):
        """
        Initialize beam with parameters from CSV file.
        
        Args:
            filename: Path to CSV file containing beam parameters
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        self.parameters = None
        self.K = None 
        self.M = None
        self.read_parameter_file(filename)
        
    def read_parameter_file(self, filename: Union[str, pathlib.Path]) -> None:
        """
        Read and validate beam parameters from CSV file.
        
        Args:
            filename: Path to CSV file containing lengths, elastic moduli,
                     moments of inertia, densities, and cross-sectional areas
            
        Raises:
            ValueError: If parameters are invalid or physically impossible
            FileNotFoundError: If file cannot be found
        """
        try:
            df = pd.read_csv(filename)
            required_cols = ['length', 'elastic_modulus', 'moment_inertia', 
                           'density', 'cross_area']
            
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")
                
            # Check for physical validity
            if (df[required_cols] <= 0).any().any():
                raise ValueError("All parameters must be positive")
                
            self.parameters = df
            # Reset matrices since parameters changed
            self.K = None
            self.M = None
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Parameter file {filename} not found")
            
    def create_stiffness_matrix(self) -> None:
        """
        Create global stiffness matrix K.
        
        Creates sparse matrix based on beam parameters and stores internally.
        """
        n_segments = len(self.parameters)
        matrix_size = 2 * (n_segments + 1)
        
        # Lists for sparse matrix construction
        rows, cols, data = [], [], []
        
        for i in range(n_segments):
            k_local = self._calculate_segment_stiffness(i)
            # Add local matrix entries to global matrix
            for local_i in range(4):
                for local_j in range(4):
                    global_i = 2*i + local_i
                    global_j = 2*i + local_j
                    rows.append(global_i)
                    cols.append(global_j)
                    data.append(k_local[local_i, local_j])
                    
        self.K = sparse.csr_matrix((data, (rows, cols)), 
                                 shape=(matrix_size, matrix_size))
        
    def create_mass_matrix(self) -> None:
        """
        Create global mass matrix M.
        
        Creates sparse matrix based on beam parameters and stores internally.
        """
        n_segments = len(self.parameters)
        matrix_size = 2 * (n_segments + 1)
        
        # Lists for sparse matrix construction
        rows, cols, data = [], [], []
        
        for i in range(n_segments):
            m_local = self._calculate_segment_mass(i)
            # Add local matrix entries to global matrix
            for local_i in range(4):
                for local_j in range(4):
                    global_i = 2*i + local_i
                    global_j = 2*i + local_j
                    rows.append(global_i)
                    cols.append(global_j)
                    data.append(m_local[local_i, local_j])
                    
        self.M = sparse.csr_matrix((data, (rows, cols)), 
                                 shape=(matrix_size, matrix_size))
    
    def get_stiffness_matrix(self) -> np.ndarray:
        """
        Return global stiffness matrix as dense matrix.
        
        Returns:
            np.ndarray: Global stiffness matrix
            
        Raises:
            RuntimeError: If matrix hasn't been created
        """
        if self.K is None:
            raise RuntimeError("Stiffness matrix not yet created")
        return self.K.toarray()
    
    def get_mass_matrix(self) -> np.ndarray:
        """
        Return global mass matrix as dense matrix.
        
        Returns:
            np.ndarray: Global mass matrix
            
        Raises:
            RuntimeError: If matrix hasn't been created
        """
        if self.M is None:
            raise RuntimeError("Mass matrix not yet created")
        return self.M.toarray()
    
    def get_length(self) -> float:
        """
        Return total length of beam.
        
        Returns:
            float: Total beam length
        """
        return self.parameters['length'].sum()
    
    def get_segment_stiffness(self, i: int) -> np.ndarray:
        """
        Return stiffness matrix for specified segment.
        
        Args:
            i: Segment index
            
        Returns:
            np.ndarray: Local stiffness matrix for segment
            
        Raises:
            IndexError: If segment index is invalid
        """
        if i < 0 or i >= len(self.parameters):
            raise IndexError(f"Segment index {i} out of range")
        return self._calculate_segment_stiffness(i)
    
    def get_segment_mass(self, i: int) -> np.ndarray:
        """
        Return mass matrix for specified segment.
        
        Args:
            i: Segment index
            
        Returns:
            np.ndarray: Local mass matrix for segment
            
        Raises:
            IndexError: If segment index is invalid
        """
        if i < 0 or i >= len(self.parameters):
            raise IndexError(f"Segment index {i} out of range")
        return self._calculate_segment_mass(i)
    
    def _calculate_segment_stiffness(self, i: int) -> np.ndarray:
        """Calculate local stiffness matrix for segment i."""
        row = self.parameters.iloc[i]
        L = row['length']
        EI = row['elastic_modulus'] * row['moment_inertia']
        
        return np.array([
            [12/(L**2),  6/L,     -12/(L**2),  6/L    ],
            [6/L,        4,       -6/L,        2      ],
            [-12/(L**2), -6/L,    12/(L**2),  -6/L    ],
            [6/L,        2,       -6/L,        4      ]
        ]) * (EI/L)
    
    def _calculate_segment_mass(self, i: int) -> np.ndarray:
        """Calculate local mass matrix for segment i."""
        row = self.parameters.iloc[i]
        L = row['length']
        rhoA = row['density'] * row['cross_area']
        
        return np.array([
            [156,    -22*L,  54,     13*L ],
            [-22*L,  4*L**2, -13*L,  -3*L**2],
            [54,     -13*L,  156,    22*L ],
            [13*L,   -3*L**2, 22*L,  4*L**2]
        ]) * (rhoA * L/420)