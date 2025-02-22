from setuptools import setup, find_packages

setup(
    name="continuum_robot",
    version="0.0.1-alpha.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.2",
    ],
)
