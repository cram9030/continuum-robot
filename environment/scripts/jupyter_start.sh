#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Create notebooks directory if it doesn't exist
mkdir -p "${PROJECT_ROOT}/notebooks"

# Build and start the Jupyter container
docker-compose -f "${PROJECT_ROOT}/environment/docker-compose.yml" up --build jupyter
