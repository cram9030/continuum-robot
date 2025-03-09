# Continuum Robot Simulation

[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Active-green.svg)](https://github.com/yourusername/continuum-robot)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simulation framework for exploring the control and path planning of continuum robots.

## 📋 Table of Contents

- [Introduction](#introduction)
- [Development Setup](#development-setup)
  - [Local Development](#local-development)
  - [Docker Development](#docker-development)
- [Running Examples](#running-examples)
- [Testing](#testing)
- [Contributing](#contributing)

## 🤖 Introduction

Continuum robots are a class of robots that lack discrete joints and instead achieve motion through continuous bending along their bodies. This project provides simulation tools for modeling, controlling, and planning paths for continuum robots, with a focus on the Euler-Bernoulli beam model implementation.

## 🛠️ Development Setup

### Local Development

#### Initial Setup

```bash
# Clone repository
git clone <repository>
cd continuum-robot

# Make setup script executable
chmod +x environment/scripts/setup.sh

# Run setup script which creates conda environment
./environment/scripts/setup.sh

# Activate environment
conda activate continuum_robot
```

#### Development Workflow

1. Launch Jupyter Lab:
   ```bash
   jupyter lab
   ```
2. Navigate to `notebooks/` directory to access development notebooks
3. Generated code will be saved to `src/notebooks/` directory

#### Environment Updates

If you update `environment.yml` or `requirements.txt`, refresh your environment with:

```bash
conda env update -f environment.yml
pip install -r requirements.txt
```

### Docker Development

Docker provides an isolated development environment with all dependencies pre-configured.

#### Running Tests

```bash
# From project root directory
cd environment
docker compose up test

# Or from any directory using -f flag
docker compose -f /path/to/continuum-robot/environment/docker-compose.yml up test
# from root
docker compose -f environment/docker-compose.yml up test
```

#### Development with Jupyter

```bash
# From project root directory
cd environment
docker compose up jupyter

# Access Jupyter Lab at http://localhost:8888
```

#### Shell Access

```bash
# Get shell in container
docker compose exec jupyter bash
```

#### Volume Mounting

The Docker setup mounts these directories for seamless development:
- `notebooks/`: Jupyter notebooks directory
- `src/`: Source code
- `tests/`: Test files

This allows editing files on your host machine while running code in the container.

### Pyodide Tests

For web-based testing with Pyodide:

```bash
# From project root
examples/pyodide_test/setup_pyode_test.sh

# Then open in browser
# http://127.0.0.1:8000/
```

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all tests
pytest tests/

# Run tests with coverage report
pytest --cov=src tests/
```

when developing new features remember to reinstall the package if the imports have changed

```
pip install -e .
```

## 🏃 Running Examples

The project includes example simulations to demonstrate functionality:

```bash
# Run linear beam simulation example
python examples/linear-beam.py
```
