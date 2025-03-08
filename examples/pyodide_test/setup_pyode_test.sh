#!/bin/bash

# Create test directory
mkdir -p dist/pyodide_test
cp dist/*.whl dist/pyodide_test/
cp examples/pyodide_test/pyodide_test.html dist/pyodide_test/index.html
