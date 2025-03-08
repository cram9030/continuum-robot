#!/bin/bash

# Create test directory
python setup.py bdist_wheel
mkdir -p dist/pyodide_test
cp dist/*.whl dist/pyodide_test/
cp examples/pyodide_test/pyodide_test.html dist/pyodide_test/index.html
python3 -m http.server --directory dist/pyodide_test 8000
