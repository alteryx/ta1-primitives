import os
import sys
from setuptools import setup, find_packages

import featuretools_ta1

MINIMUM_PYTHON_VERSION = 3, 6


# Exit when the Python version is too low.
if sys.version_info < MINIMUM_PYTHON_VERSION:
    sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


d3m_primitives = list()
for primitive in featuretools_ta1.PRIMITIVES:
    python_path = primitive.metadata.query()['python_path']
    name = python_path[15:]   # remove the d3m.primitives part
    entry_point = '{} = {}:{}'.format(name, primitive.__module__, primitive.__name__)
    d3m_primitives.append(entry_point)


setup(
    name='featuretools_ta1',
    version=featuretools_ta1.__version__,
    description='Primitives using Featuretools, an open source feature engineering platform',
    author=featuretools_ta1.__author__,
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'd3m==2019.1.21',
        'featuretools==0.5.1',
    ],
    url='https://gitlab.datadrivendiscovery.org/MIT-FeatureLabs/ta1-primitives',
    entry_points={
        'd3m.primitives': d3m_primitives,
    },
)
