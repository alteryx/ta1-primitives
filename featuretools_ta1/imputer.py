from typing import Dict
import os
from sklearn.preprocessing.imputation import Imputer as SKImputer
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.container.pandas import DataFrame
from .utils import get_target_columns
import numpy as np
import pandas as pd

from . import __version__

ALL_ELEMENTS = metadata_module.ALL_ELEMENTS

Inputs = DataFrame
Outputs = DataFrame


class Hyperparams(hyperparams.Hyperparams):
    strategy = hyperparams.Enumeration[str](
        default='mean',
        values=['median', 'most_frequent', 'mean'],
        description='The imputation strategy.  - If "mean", then replace missing values using the mean along the axis. - If "median", then replace missing values using the median along the axis. - If "most_frequent", then replace missing using the most frequent value along the axis. ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    exclude_target = hyperparams.Hyperparameter[bool](
        default=True,
        description='''
Exclude target column from imputation''',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])


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
                 docker_containers: Dict[str, DockerContainer] = None,
                 _verbose: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._strategy = hyperparams['strategy']
        self._exclude_target = hyperparams['exclude_target']
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
        use_cols = inputs.columns
        if self._exclude_target:
            target_cols = get_target_columns(inputs.metadata)
            use_cols = [c for c in inputs if c not in target_cols]
        old_metadata = inputs.metadata
        X = inputs[use_cols].astype(np.float32)
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
        if self._exclude_target:
            for c in target_cols:
                output[c] = inputs[c].values
        output.metadata = self._update_metadata(old_metadata, output)
        return CallResult(output)

    def _update_metadata(self, old_metadata, df):
        new_metadata = old_metadata

        for i, c in enumerate(df.columns):
            existing = old_metadata.query((ALL_ELEMENTS, i))
            new = {k: v for k, v in existing.items()}
            new['structural_type'] = type(df[c].iloc[0])
            new_metadata = new_metadata.update((ALL_ELEMENTS, i),
                                               new)
        return new_metadata
