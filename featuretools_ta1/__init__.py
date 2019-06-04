__version__ = '0.4.0'
__author__ = 'MIT/Feature Labs Team'


from featuretools_ta1.dfs import DFS
from featuretools_ta1.single_table import SingleTableFeaturization

PRIMITIVES = [
    DFS,
    SingleTableFeaturization
]
