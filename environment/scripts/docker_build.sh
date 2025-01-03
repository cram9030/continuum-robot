#!/bin/bash
# build_docker.sh

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Build the Docker image
docker build \
  -t continuum_robot \
  -f "${PROJECT_ROOT}/environment/docker/Dockerfile" \
  "${PROJECT_ROOT}"
