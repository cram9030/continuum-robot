#!/bin/bash
# build_docker.sh

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Build the base image
docker build \
  -t continuum_robot \
  -f "${PROJECT_ROOT}/environment/docker/Dockerfile" \
  "${PROJECT_ROOT}"

# Build the Jupyter image
docker build \
  -t continuum_robot_jupyter \
  -f "${PROJECT_ROOT}/environment/docker/Dockerfile.jupyter" \
  "${PROJECT_ROOT}"
