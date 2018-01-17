from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Dict, Union
from d3m_metadata.container.dataset import Dataset
from collections import OrderedDict
from d3m_metadata.container import List
from d3m_metadata.container.pandas import DataFrame
from d3m_metadata import (hyperparams, params,
                          metadata as metadata_module, utils)
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from primitive_interfaces.base import CallResult
from featuretools import primitives as ftypes
from itertools import combinations, chain
import os

import featuretools as ft
from featuretools import variable_types as vtypes
from .d3m_to_entityset import convert_d3m_dataset_to_entityset
import pandas as pd
# from inspect import getargspec
# import copy

# First element is D3MDataset, second element is dict of a target from problemDoc.json
Input = List[Union[Dataset, dict]]

# TODO: make output another D3MDS?
Output = DataFrame # Featurized dataframe, indexed by the same index as Input. Features (columns) have human-readable names

# This is a class representing the internal state of the primitive.
# Notice the colon syntax, mapping the name to the type
class Params(params.Params):
    # List of featuretools.PrimitiveBase objects representing the features. Serializable via ft.save_features(feature_list, file_object)
    # A named tuple for parameters.
    features: List[object]


# This is a custom hyperparameter type. You may find it useful if you have a list/set of
# options, and the user can select any number of them
class SetHyperparam(hyperparams.Enumeration[object]):
    def __init__(self, options, default=None, description=None, max_to_remove=3):
        lower_limit = len(options) - max_to_remove
        upper_limit = len(options) + 1
        if default is None:
            default = set(options)
        else:
            default = set(default)

        sets = list(chain(*[list([set(o) for o in combinations(options, i)])
                            for i in range(lower_limit, upper_limit)]))
        super().__init__(values=sets, default=default,
                         description=description)


# Hyperparams need to be defined as a new class, because everything is strongly typed
# Notice the equals syntax (different than the colon syntax in Params)
# For more type definitions, see https://gitlab.com/datadrivendiscovery/metadata/blob/devel/d3m_metadata/hyperparams.py
# For more examples, see https://gitlab.datadrivendiscovery.org/jpl/d3m_sklearn_wrap
class Hyperparams(hyperparams.Hyperparams):
    d = OrderedDict()
    d['specified'] = hyperparams.UniformInt(
                                  lower=1,
                                  upper=5,
                                  default=2,
                                  description=''
                            )
    d['any'] = hyperparams.UniformInt(
                                lower=-1,
                                upper=0,
                                default=-1
                            )
    max_depth = hyperparams.Union(d, default='specified',
                                  description='maximum allowed depth of features')
    normalize_categoricals_if_single_table = hyperparams.Hyperparameter[bool](
        default=True,
        description='If dataset only has a single table and normalize_categoricals_if_single_table is True, then normalize categoricals into separate entities.'
    )

    agg_primitive_options = [ftypes.Sum, ftypes.Std, ftypes.Max, ftypes.Skew,
                             ftypes.Min, ftypes.Mean, ftypes.Count,
                             ftypes.PercentTrue, ftypes.NUnique, ftypes.Mode,
                             ftypes.Trend]
    default_agg_prims = [ftypes.Sum, ftypes.Std, ftypes.Max, ftypes.Skew,
                         ftypes.Min, ftypes.Mean, ftypes.Count,
                         ftypes.PercentTrue, ftypes.NUnique, ftypes.Mode]

    agg_primitives = SetHyperparam(
        options=agg_primitive_options,
        default=default_agg_prims,
        max_to_remove=4,
        description='list of Aggregation Primitives to apply.'
    )
    trans_primitive_options = [ftypes.Day, ftypes.Year, ftypes.Month,
                               ftypes.Days, ftypes.Years, ftypes.Months,
                               ftypes.Weekday, ftypes.Weekend,
                               ftypes.TimeSince,
                               ftypes.Percentile]

    default_trans_prims = [ftypes.Day, ftypes.Year, ftypes.Month, ftypes.Weekday]
    trans_primitives = SetHyperparam(
        options=trans_primitive_options,
        max_to_remove=6,
        description='list of Transform Primitives to apply.'
    )



# See https://gitlab.com/datadrivendiscovery/primitive-interfaces
# for all the different possible primitive types to subclass from
class DFS(UnsupervisedLearnerPrimitiveBase[Input, Output, Params, Hyperparams]):
    """
    Primitive wrapping featuretools on single table datasets
    """
    __author__ = 'Feature Labs D3M team (Ben Schreck <ben.schreck@featurelabs.com>)'

    # For a list of options for each of these fields, see
    # https://metadata.datadrivendiscovery.org/
    metadata = metadata_module.PrimitiveMetadata(
        {'algorithm_types': ['DEEP_FEATURE_SYNTHESIS', ],
         'name': 'Deep Feature Synthesis',
         'primitive_family': 'FEATURE_CONSTRUCTION',
         'python_path': 'd3m.primitives.featuretools_ta1.DFS',
         "source": {
           "name": "MIT_FeatureLabs",
           "contact": "mailto://ben.schreck@featurelabs.com",
           "uris": ["https://doc.featuretools.com"],
           "license": "BSD-3-Clause"

         },
         "description": "Calculates a feature matrix and features given a single-table tabular D3M Dataset.",
         "keywords": ["featurization", "feature engineering", "feature extraction"],
         "hyperparameters_to_tune": ["max_depth", "normalize_categoricals_if_single_table"],
         'version': '0.1.0',
         'id': 'c4cd2401-6a66-4ddb-9954-33d5a5b61c52',
         'installation': [{'type': metadata_module.PrimitiveInstallationType.PIP,
                           'package_uri': 'git+https://github.com/Featuretools/ta1-primitives.git@{git_commit}#egg=featuretools_ta1'.format(
                               git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                            ),
                          }]
        })

    # Output type for this needs to be specified (and should be None)
    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, str] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed,
                         docker_containers=docker_containers)
        # All saved attributes must be prefixed with underscores
        # Can treat hyperparams as a normal dict
        self._max_depth = hyperparams['max_depth']
        self._normalize_categoricals_if_single_table = \
            hyperparams['normalize_categoricals_if_single_table']
        self._agg_primitives = list(hyperparams['agg_primitives'])
        self._trans_primitives = list(hyperparams['trans_primitives'])

        # Initialize all the attributes you will eventually save
        self._target_entity = None
        self._target = None
        self._entityset = None
        self._features = None
        self._fitted = False

    # Output type for this needs to be specified (and should be None)
    def set_training_data(self, *, inputs: Input) -> None:
        parsed = self._parse_inputs(inputs)
        self._entityset, self._target_entity, self._target, self._entities_normalized, _ = parsed
        self._fitted = False

    def _parse_inputs(self, inputs, entities_to_normalize=None,
                      original_entityset=None):
        target = inputs[1]
        if 'colName' in target:
            target['column_name'] = target['colName']
            del target['colName']
        entityset, target_entity, entities_normalized, instance_ids = convert_d3m_dataset_to_entityset(
                inputs[0],
                entities_to_normalize=entities_to_normalize,
                original_entityset=original_entityset,
                normalize_categoricals_if_single_table=self._normalize_categoricals_if_single_table)
        return entityset, target_entity, target, entities_normalized, instance_ids

    # Output type for this needs to be specified (and should be None)
    def set_params(self, *, params: Params) -> None:
        self._features = params

    # Output type for this needs to be specified (and should be Params)
    def get_params(self) -> Params:
        return Params(features=self._features)

    # Output type for this needs to be specified (and should be CallResult[None])
    def fit(self, *, timeout: float=None, iterations: int=None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._entityset is None:
            raise ValueError("Must call .set_training_data() before calling .fit()")
        ignore_variables = {self._target_entity: [self._target['column_name']]}
        time_index = self._entityset[self._target_entity].time_index
        index = self._entityset[self._target_entity].index
        cutoff_time = None
        if time_index:
            cutoff_time = self._entityset[self._target_entity].df[[index, time_index]]
        self._features = ft.dfs(entityset=self._entityset,
                                target_entity=self._target_entity,
                                cutoff_time=cutoff_time,
                                features_only=True,
                                ignore_variables=ignore_variables,
                                max_depth=self._max_depth,
                                agg_primitives=self._agg_primitives,
                                trans_primitives=self._trans_primitives)
        return CallResult(None)

    # Output type for this needs to be specified (and should be CallResult[Output])
    def produce(self, *, inputs: Input, timeout: float=None, iterations: int=None) -> CallResult[Output]:
        if self._features is None:
            raise ValueError("Must call fit() before calling produce()")
        features = self._features

        parsed = self._parse_inputs(inputs,
                                    entities_to_normalize=self._entities_normalized,
                                    original_entityset=self._entityset)
        entityset, target_entity, target, _, instance_ids = parsed

        feature_matrix = ft.calculate_feature_matrix(features,
                                                     entityset=entityset,
                                                     instance_ids=instance_ids,
                                                     cutoff_time_in_index=True)

        feature_matrix = (feature_matrix.reset_index('time')
                                        .loc[instance_ids, :]
                                        .set_index('time', append=True))
        for f in features:
            if issubclass(f.variable_type, vtypes.Discrete):
                feature_matrix[f.get_name()] = feature_matrix[f.get_name()].astype(object)
            elif issubclass(f.variable_type, vtypes.Numeric):
                feature_matrix[f.get_name()] = pd.to_numeric(feature_matrix[f.get_name()])
            elif issubclass(f.variable_type, vtypes.Datetime):
                feature_matrix[f.get_name()] = pd.to_datetime(feature_matrix[f.get_name()])
        return CallResult(feature_matrix)
