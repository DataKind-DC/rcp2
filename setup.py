from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Project objective',
    author='DataKindDC',
    license='MIT',
    install_requires=[
        "geopandas>=0.8.0",
        "pandas>=1.0.5",
        "Shapely>=1.7.0",
    ],
    extras_require={
        "tests": "pytest>=5.4.3",
    },
)
