# continuum-robot
Simulation for exploring the control and path planning of continuum robots


# Development Setup

## Local Development Setup

### Initial setup

```
# Clone repository
git clone <repository>
cd continuum-robot

# Make setup script executable
chmod +x environment/scripts/setup.sh

# Run setup script which creates conda environment
./environment/scripts/setup.sh

# Activate environment
conda activate continuum_robot

# Launch Jupyter Lab
jupyter lab
```

### Development Workflow

1. Jupyter Lab will open in your default browser at http://localhost:8888
2. Navigate to notebooks/ directory to access development notebooks
3. Generated code will be saved to src/notebooks/ directory

#### Runing Tests
Run tests using:
```
pytest tests/
```

### Notes on Environment Updates

If you update environment.yml or requirements.txt, you can update your environment with:
```
conda env update -f environment.yml
pip install -r requirements.txt
```

## Docker Usage

### Running Tests

```
# From project root directory
cd environment
docker compose up test

# Or from any directory using -f flag
docker compose -f /path/to/continuum-robot/environment/docker-compose.yml up test
# from root
docker compose -f environment/docker-compose.yml up test
```

### Development with Jupyter
```
# From project root directory
cd environment
docker compose up jupyter

# Or from any directory using -f flag
docker compose -f /path/to/continuum-robot/environment/docker-compose.yml up jupyter

# Access Jupyter Lab at http://localhost:8888
```

### Shell Access
```
# Get shell in container
docker compose exec jupyter bash
```

### Note on Volumes
The Docker setup mounts these directories:

notebooks/: Jupyter notebooks directory
src/: Source code
tests/: Test files

This allows editing files on your host machine while running code in the container.
