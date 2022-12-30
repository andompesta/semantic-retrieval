"""
This file configures the Python package with entrypoints used for future runs on Databricks.

Please follow the `entry_points` documentation for more details on how to configure the entrypoint:
* https://setuptools.pypa.io/en/latest/userguide/entry_point.html
"""

from setuptools import find_packages, setup
from semantic_retrieval import __version__

PACKAGE_REQUIREMENTS = [
    "ftfy",
    "comet-ml",
]

DEV_REQUIREMENTS = [
    # installation & build
    "setuptools",
    "wheel",
    # versions set in accordance with DBR 10.4 ML Runtime
    "pyspark==3.2.1",
    "delta-spark==1.1.0",
    # generic dependencies
    "pyyaml",
    "Pillow",
    "scikit-learn",
    "pandas",
    "mlflow",
    # development & testing tools
    "pytest",
    "pytest-cov",
    "dbx>=0.8",
    "petastorm",
    "torch",
    "torchvision",
    "boto3"
]

setup(
    name="semantic_retrieval",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    install_requires=PACKAGE_REQUIREMENTS,
    extras_require={"dev": DEV_REQUIREMENTS},
    version=__version__,
    description="",
    author="",
)
