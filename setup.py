import os
import sys
from setuptools import setup, find_packages

MINIMUM_PYTHON_VERSION = 3, 6


# Exit when the Python version is too low.
if sys.version_info < MINIMUM_PYTHON_VERSION:
    sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


d3m_primitives = [
    'feature_construction.deep_feature_synthesis.Featuretools = featuretools_ta1.dfs:DFS'
]


setup(
    author='MIT/Feature Labs Team',
    description='Primitives using Featuretools, an open source feature engineering platform',
    name='featuretools_ta1',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'd3m==2019.2.18',
        'featuretools==0.6.1',
    ],
    url='https://gitlab.datadrivendiscovery.org/MIT-FeatureLabs/ta1-primitives',
    entry_points={
        'd3m.primitives': d3m_primitives,
    },
    version='0.3.3-dev',
)
