__version__ = '0.3.4-dev'
__author__ = 'MIT/Feature Labs Team'


from featuretools_ta1.dfs import DFS
from featuretools_ta1.single_table import SingleTableDFS

PRIMITIVES = [
    DFS,
    SingleTableDFS
]
