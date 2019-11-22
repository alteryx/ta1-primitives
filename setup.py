import os
import sys
from setuptools import setup, find_packages

MINIMUM_PYTHON_VERSION = 3, 6


# Exit when the Python version is too low.
if sys.version_info < MINIMUM_PYTHON_VERSION:
    sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


d3m_primitives = [
    'feature_construction.deep_feature_synthesis.SingleTableFeaturization = featuretools_ta1.single_table:SingleTableFeaturization',
    'feature_construction.deep_feature_synthesis.MultiTableFeaturization = featuretools_ta1.multi_table:MultiTableFeaturization'
]


setup(
    author='MIT/Feature Labs Team',
    description='Primitives using Featuretools, an open source feature engineering platform',
    name='featuretools_ta1',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'd3m',
        'featuretools @ git+https://github.com/featuretools/featuretools.git@0bef1727b190126c230264d158aa6d3480dd0a9d#egg=featuretools',
    ],
    url='https://gitlab.datadrivendiscovery.org/MIT-FeatureLabs/ta1-primitives',
    entry_points={
        'd3m.primitives': d3m_primitives,
    },
    version='0.7.0',
)
