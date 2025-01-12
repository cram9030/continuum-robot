#!/bin/bash

# Function to initialize conda
init_conda() {
    echo "Initializing conda..."
    CONDA_PATH="$HOME/miniconda3/bin/conda"

    # Initialize conda for current shell
    eval "$($CONDA_PATH shell.bash hook)"

    # Add conda to PATH for current session
    export PATH="$HOME/miniconda3/bin:$PATH"

    # Initialize conda in shell startup
    $CONDA_PATH init bash
}

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Check if miniconda directory exists
if [ -d "$HOME/miniconda3" ]; then
    echo "Miniconda directory found..."

    # Check if conda command is available
    if ! command -v conda &> /dev/null; then
        echo "Conda command not found. Setting up path..."
        init_conda
    fi
else
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh
    init_conda
fi

# Verify conda is now available
if command -v conda &> /dev/null; then
    echo "Conda installation verified."

    # Create conda environment
    echo "Creating conda environment..."
    conda env create -f "${PROJECT_ROOT}/environment.yml"

    # Activate environment
    echo "Activating environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate continuum_robot

    # Install additional dependencies
    echo "Installing additional dependencies..."
    pip install -r "${PROJECT_ROOT}/requirements.txt"

    # Install Jupyter Lab extensions
    echo "Installing Jupyter Lab extensions..."
    jupyter labextension install @jupyterlab/toc
    jupyter labextension install @jupyterlab/git

    # Setup pre-commit hooks
    echo "Setting up pre-commit hooks..."
    pip install pre-commit
    pre-commit install
else
    echo "Error: Conda installation failed or PATH not updated correctly"
    echo "Please try running: source ~/.bashrc"
    echo "Or restart your terminal session"
    exit 1
fi

echo "Setup complete! Please run 'source ~/.bashrc' or restart your terminal,"
echo "then use 'conda activate continuum_robot' to start working."

# Create notebook directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/notebooks"
