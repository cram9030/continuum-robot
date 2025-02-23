# Dockerfile
FROM ubuntu:24.04 as builder

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up conda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Set up working directory
WORKDIR /app

# Create minimal conda environment for building
RUN conda create -n test python=3.10 pip -y

# Copy only what's needed for building the wheel
COPY setup.py ./
COPY src/ ./src/

# Install build tools and build the wheel
RUN conda run -n test pip install build wheel && \
    conda run -n test python -m build --wheel

# Start a fresh image for testing
FROM ubuntu:24.04 as tester

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3-dev \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Set up conda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

WORKDIR /app

# Create minimal test environment with just Python
RUN conda create -n test python=3.10 pip -y

# Copy the wheel and test files from builder
COPY --from=builder /app/dist/*.whl ./
COPY pytest.ini ./
COPY tests/ ./tests/

# Install the wheel and test dependencies in the minimal environment
RUN conda run -n test pip install *.whl pytest pytest-cov

# Run tests by default using the minimal conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "test", "pytest", "tests/"]
