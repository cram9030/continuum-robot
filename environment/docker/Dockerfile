# Base image
FROM ubuntu:24.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set up timezone data
RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3-dev \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Add conda to path
ENV PATH=$CONDA_DIR/bin:$PATH

# Copy environment files
COPY environment.yml requirements.txt ./

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "continuum_robot", "/bin/bash", "-c"]

# Install additional pip requirements if needed
RUN pip install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY pytest.ini ./

# Set Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Activate conda environment by default
RUN echo "conda activate continuum_robot" >> ~/.bashrc

# Default command to run tests using the conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "continuum_robot", "pytest", "tests/"]
