# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
- `python -m pytest tests/` - Run all tests
- `python -m pytest tests/test_linear_euler_bernoulli_beam.py -v` - Run specific test file with verbose output
- `python -m pytest tests/test_linear_euler_bernoulli_beam.py::test_stiffness_matrix -v` - Run single test
- `python -m pytest tests/ --tb=short` - Run tests with short traceback format

### Linting and Code Quality
- Code is automatically linted via pre-commit hooks using `black` and `flake8`
- Pre-commit hooks run on every commit and handle formatting

## Architecture Overview

### Core Beam Models
The library implements finite element continuum robot modeling with two primary beam formulations:

**Unified 3-DOF Node Structure**: Both linear and nonlinear models now use 3 degrees of freedom per node:
- `u`: Axial displacement
- `w`: Transverse displacement
- `φ` (phi): Rotation

**DOF Organization**: State vectors are organized as `[u1, w1, φ1, u2, w2, φ2, ...]`

### Key Classes

**EulerBernoulliBeam** (`src/continuum_robot/models/euler_bernoulli_beam.py`):
- Unified beam implementation supporting hybrid beams with mixed linear and nonlinear segments
- Automatically detects segment types from 'type' column in parameter DataFrame
- Uses SegmentFactory to create appropriate segment types (linear or nonlinear)
- 6×6 segment matrices with 3-DOF nodes (u, w, φ per node)
- Supports boundary conditions: FIXED (constrains u,w,φ) and PINNED (constrains u,w)
- DOF mapping system maintains correspondence between DOF indices and (parameter, node) pairs
- Provides both stiffness function (for general/hybrid beams) and stiffness matrix (linear-only beams)
- Key methods: `get_stiffness_function()`, `get_stiffness_matrix()`, `apply_boundary_conditions()`

**DynamicEulerBernoulliBeam** (`src/continuum_robot/models/dynamic_beam_model.py`):
- Dynamic wrapper around unified EulerBernoulliBeam for time-domain simulation
- State vector: first half positions, second half velocities
- Supports auto-registration of force components (fluid drag, gravity) based on ForceParams
- Force and input registries for modular force/control system composition
- Provides system function for solve_ivp integration with external force inputs
- State mapping system tracks correspondence between state indices and physical quantities

**Control System Architecture**:

**AbstractInputHandler** (`src/continuum_robot/control/control_abstractions.py`):
- Abstract base class for input processing components
- Interface: `compute_input(x, r, t)` returns input modifications based on state, reference, and time
- Enables modular control system composition through InputRegistry

**FullStateLinear** (`src/continuum_robot/control/full_state_linear.py`):
- Implements state feedback control: `u_modified = u - K * (x - x_ref)`
- Takes gain matrix K for linear state feedback applications
- Useful for LQR and other linear control strategies

**LinearQuadraticRegulator** (`src/control_design/linear_quadratic_regulator.py`):
- LQR controller design for linear continuum robot beams
- Computes optimal gain matrices from beam stiffness/mass matrices and Q/R weighting
- Provides A and B matrices for state-space representation
- Validates system stability after gain computation

**Registry System** (`src/continuum_robot/models/force_registry.py`):

**ForceRegistry**: Manages force components independently from beam dynamics
- Registers AbstractForce implementations (fluid drag, gravity, etc.)
- Creates aggregated force functions for dynamic simulations
- Supports enable/disable and modular composition

**InputRegistry**: Manages input processing components independently
- Registers AbstractInputHandler implementations for control system composition
- Creates aggregated input processing functions
- Enables layered control architectures (feedforward + feedback + disturbance rejection)

### Critical Implementation Details

**DOF Mapping System**: Both beam models maintain mappings between DOF indices and (parameter, node) pairs. These mappings are dynamically updated when boundary conditions are applied.

**Boundary Condition Handling**:
- Creates dimension-reduced systems by eliminating constrained DOFs
- Updates both matrices and DOF mappings consistently
- Supports clearing/reapplying boundary conditions

**State Vector Extraction in Examples**:
- Linear/nonlinear beams now both use `sol.y[n_pos + 1::3, i]` to extract transverse displacements
- Examples include fluid dynamics parameter columns (can be set to 0 to disable)

### Testing Architecture
Tests use temporary CSV files with Nitinol material properties and validate:
- Matrix dimensions (15×15 for 4-segment beam with 5 nodes × 3 DOFs)
- Boundary condition applications and DOF reductions
- DOF mapping consistency before/after boundary conditions
- Numerical properties (symmetry, positive definiteness)

### Development Patterns
- All CSV parameter files require: `length,elastic_modulus,moment_inertia,density,cross_area`
- Dynamic beam CSV files additionally require: `type,boundary_condition,wetted_area,drag_coef`
- Segment stiffness/mass matrices are 6×6 for 3-DOF nodes
- Global matrices assembled using sparse matrix operations for efficiency

### State Vector Indexing
When extracting beam shapes from simulation results:
- Position states: first half of state vector
- Velocity states: second half of state vector
- Transverse displacements (`w`): indices `1, 4, 7, 10, ...` (pattern: `1 + 3*j`)
