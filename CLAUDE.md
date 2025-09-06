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

**LinearEulerBernoulliBeam** (`src/continuum_robot/models/linear_euler_bernoulli_beam.py`):
- 6×6 segment matrices with linear stiffness `EA/L` for axial terms
- Mass matrix includes axial terms following pattern: `140*ρA*L/420` for axial components
- Supports boundary conditions: FIXED (constrains u,w,φ) and PINNED (constrains u,w)

**NonlinearEulerBernoulliBeam** (`src/continuum_robot/models/nonlinear_euler_bernoulli_beam.py`):
- Uses complex nonlinear stiffness functions with geometric coupling between axial and transverse deformations
- MapReduce architecture for parallel segment force computation
- Pre-computed segment stiffness functions for performance

**DynamicEulerBernoulliBeam** (`src/continuum_robot/models/dynamic_beam_model.py`):
- Integrates linear/nonlinear models for time-domain simulation
- State vector: first half positions, second half velocities
- Supports fluid dynamics with drag forces on transverse velocities
- Mixed linear/nonlinear systems not currently supported

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
