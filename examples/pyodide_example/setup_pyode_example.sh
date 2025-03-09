#!/bin/bash

# Create test directory
python setup.py bdist_wheel
mkdir -p dist/pyodide_example
cp dist/*.whl dist/pyodide_example/
cp examples/pyodide_example/pyodide_example.py dist/pyodide_example/
cp examples/pyodide_example/pyodide_example.html dist/pyodide_example/index.html
python3 -m http.server --directory dist/pyodide_example 8000
