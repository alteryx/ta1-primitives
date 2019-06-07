from d3m.metadata import base as metadata_base, hyperparams, params
from d3m import container, exceptions
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from typing import Dict, Optional, Union
from featuretools_ta1 import config as CONFIG
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.exceptions import PrimitiveNotFittedError
from featuretools_ta1.utils import drop_percent_null, select_one_of_correlated
import featuretools_ta1
from featuretools_ta1.utils import get_featuretools_variable_types, find_primary_key, add_metadata
import featuretools as ft
import numpy as np
import pandas as pd

Inputs = container.DataFrame
Outputs = container.DataFrame
TARGET_ENTITY = "table"

class Params(params.Params):
    # A named tuple for parameters.
    features: Optional[bytes]

class Hyperparams(hyperparams.Hyperparams):
    max_percent_null = hyperparams.Bounded[float](
        default=.5,
        lower=0,
        upper=1,
        description='The maximum percentage of null values allowed in returned features. A lower value means features may have more null nulls.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_correlation = hyperparams.Bounded[float](
        default=.9,
        lower=0,
        upper=1,
        description='The maximum allowed correlation between any two features returned. A lower value means features will be more uncorrelated',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='new',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.",
    )


class SingleTableFeaturization(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """This primitive creates new interaction features for an input dataframe.

    After creating features it reduces the set of possible features using an unsupervised approach"""
    __author__ = 'Max Kanter <max.kanter@featurelabs.com>'
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '6c5dcfa3-1f87-4066-b16a-88c9c971f6e3',
            'version': featuretools_ta1.__version__,
            'name': "Single Table Deep Feature Synthesis",
            'python_path': 'd3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization',
            'source': {
                'name': CONFIG.AUTHOR,
                'contact': CONFIG.CONTACT,
                'uris': ['https://docs.featuretools.com'],
                'license': 'BSD-3-Clause'
            },
            'installation': CONFIG.INSTALLATION,
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DEEP_FEATURE_SYNTHESIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_CONSTRUCTION,
            'keywords': [
                'featurization',
                'feature engineering',
                'feature extraction',
                'feature construction'
            ],
            'hyperparameters_to_tune': ['max_percent_null', 'max_correlation'],
        },
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        self._fitted = False

        # todo handle use_columns, exclude_columns
        # todo handle return result

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._input_df = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        es = self._make_entityset(self._input_df)


        trans_primitives = ["add_numeric", "subtract_numeric", "multiply_numeric", "divide_numeric",
                            "is_weekend", "day", "month", "year", "week", "weekday"]

        # generate all the features
        fm, features = ft.dfs(
            target_entity=TARGET_ENTITY,
            entityset=es,
            agg_primitives=[],
            trans_primitives=trans_primitives,
            max_depth=1,
        )

        # treat inf as null. repeat in produce step
        fm = fm.replace([np.inf, -np.inf], np.nan)

        # filter based on nulls and correlation
        fm, features = drop_percent_null(fm, features, max_percent_null=self.hyperparams['max_percent_null'])
        fm, features = select_one_of_correlated(fm, features, threshold=self.hyperparams['max_correlation'])

        self.features = features

        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        es = self._make_entityset(inputs)

        fm = ft.calculate_feature_matrix(
            entityset=es,
            features=self.features
        )

        fm = fm.reindex(es[TARGET_ENTITY].df.index)

        # treat inf as null like fit step
        fm = fm.replace([np.inf, -np.inf], np.nan)

        fm = add_metadata(fm, self.features)

        return CallResult(fm)

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(features=None)

        return Params(features=None)

    def set_params(self, *, params: Params) -> None:
        self.features = params["features"]

        # infer if it is fitted
        if self.features:
            self._fitted = True


    def _make_entityset(self, input_df):
        es = ft.EntitySet()

        primary_key = find_primary_key(input_df)

        if primary_key is None:
            primary_key = "D3M_INDEXx"

        variable_types = get_featuretools_variable_types(input_df)

        es.entity_from_dataframe(entity_id=TARGET_ENTITY,
                                 dataframe=pd.DataFrame(input_df.copy()),
                                 index=primary_key,
                                 make_index=True,
                                 variable_types=variable_types)

        return es


"""

do we need the docker containers input to init?

# def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
fix path

TODOS
* handle non numeric
"""
