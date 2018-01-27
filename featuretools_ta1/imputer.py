from typing import Union, Dict
import os
from sklearn.preprocessing.imputation import Imputer as SKImputer
from d3m_metadata.container.numpy import ndarray
from d3m_metadata import hyperparams, metadata as metadata_module, utils
from primitive_interfaces.transformer import TransformerPrimitiveBase
from primitive_interfaces.base import CallResult
from d3m_metadata.container.pandas import DataFrame
import numpy as np
import pandas as pd

from . import __version__

Inputs = Union[ndarray, DataFrame]
# If passed a DataFrame, will output a DataFrame
# If passed an ndarray, will output an ndarray
Outputs = Union[ndarray, DataFrame]


class Hyperparams(hyperparams.Hyperparams):
    strategy = hyperparams.Enumeration[str](
        default='mean',
        values=['median', 'most_frequent', 'mean'],
        description='The imputation strategy.  - If "mean", then replace missing values using the mean along the axis. - If "median", then replace missing values using the median along the axis. - If "most_frequent", then replace missing using the most frequent value along the axis. '
    )


class Imputer(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Wrap sklearn's Imputer to make sure it
    does not drop any features which end
    up being all nan in the cross-val split
    """

    __author__ = 'Feature Labs D3M team (Ben Schreck <ben.schreck@featurelabs.com>)'
    metadata = metadata_module.PrimitiveMetadata({
         "algorithm_types": ['IMPUTATION'],
         "name": "Robust Imputer",
         "primitive_family": "DATA_PREPROCESSING",
         "python_path": "d3m.primitives.featuretools_ta1.Imputer",
         "source": {
           "name": "MIT_FeatureLabs",
           "contact": "mailto://ben.schreck@featurelabs.com",
           "license": "BSD-3-Clause"

         },
         "description": """
Wrap sklearn's Imputer to make sure it
does not drop any features which end
up being all nan in the cross-val split
""",
         "version": __version__,
         "id": '2c3d9077-c8cd-4497-94a3-c38614138fc8',
         'installation': [{'type': metadata_module.PrimitiveInstallationType.PIP,
                           'package_uri': 'git+https://github.com/Featuretools/ta1-primitives.git@{git_commit}#egg=featuretools_ta1-{version}'.format(
                               git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                               version=__version__
                            ),
                          }]

    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, str] = None,
                 _verbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._strategy = hyperparams['strategy']
        self._imputer_nan = SKImputer(
            missing_values='NaN',
            strategy=self._strategy,
            axis=0,
            verbose=_verbose
        )
        self._imputer_inf = SKImputer(
            missing_values=np.inf,
            strategy=self._strategy,
            axis=0,
            verbose=_verbose
        )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        input_was_ndarray = isinstance(inputs, np.ndarray)

        X = inputs.astype(np.float32)
        df = pd.DataFrame(X).copy()
        all_nans = []
        other_columns = []
        for i, c in enumerate(df):
            if df[c].dropna().shape[0] == 0:
                all_nans.append(c)
            else:
                other_columns.append(c)

        df[all_nans] = 0.0
        imputed = self._imputer_nan.fit_transform(X)
        imputed2 = self._imputer_inf.fit_transform(imputed)
        df[other_columns] = imputed2

        output = df
        if input_was_ndarray:
            output = output.values
        return CallResult(output)
