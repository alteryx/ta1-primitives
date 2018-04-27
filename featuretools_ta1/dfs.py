from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Dict, Union, Optional
import typing
from .utils import serialize_features, load_features
from d3m.container.dataset import Dataset
from collections import OrderedDict
from d3m.container import List
from d3m.container.pandas import DataFrame
from d3m.metadata import hyperparams, params, base as metadata_module
from d3m import utils
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from itertools import combinations, chain
import cloudpickle
import os

import featuretools as ft
from featuretools import variable_types as vtypes
from featuretools.selection import remove_low_information_features
from .d3m_to_entityset import convert_d3m_dataset_to_entityset
import pandas as pd
from . import __version__

# First element is D3MDataset, second element is dict of a target from problemDoc.json
Input = List[Union[Dataset, dict]]

# TODO: maybe make output another D3MDS?
Output = DataFrame # Featurized dataframe, indexed by the same index as Input. Features (columns) have human-readable names
EncodedOutput = List[Union[DataFrame, List]]

# This is a class representing the internal state of the primitive.
# Notice the colon syntax, mapping the name to the type
class Params(params.Params):
    # List of featuretools.PrimitiveBase objects representing the features. Serializable via ft.save_features(feature_list, file_object)
    # A named tuple for parameters.
    entityset: ft.EntitySet
    fitted: bool
    target_entity: str
    target: dict
    entities_normalized: bytes
    features: Optional[bytes]

def sort_by_str(l):
    return sorted(l, key=lambda x: str(x))


# This is a custom hyperparameter type. You may find it useful if you have a list/set of
# options, and the user can select any number of them
class ListHyperparam(hyperparams.Enumeration[object]):
    def __init__(self, options, default=None, description=None, max_to_remove=3):
        lower_limit = len(options) - max_to_remove
        upper_limit = len(options) + 1
        if default is None:
            default = sort_by_str(options)
        else:
            default = sort_by_str(default)

        lists = list(chain(*[list([sort_by_str(o) for o in combinations(options, i)])
                            for i in range(lower_limit, upper_limit)]))
        super().__init__(values=lists, default=default,
                         description=description)


T = typing.TypeVar('FTPrimitive')


class GenericListHyperparam(hyperparams.Hyperparameter):
    def __init__(self, default, description=None):
        self.structural_type = typing.List[T]
        super().__init__(default=default,
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
    find_equivalent_categories = hyperparams.Hyperparameter[bool](
        default=True,
        description=''
    )
    d = OrderedDict()
    d['fraction'] = hyperparams.Uniform(
                                  lower=0.00001,
                                  upper=1,
                                  default=.1,
                                  description='fraction of nunique values'
                            )
    d['value'] = hyperparams.UniformInt(
                                lower=1,
                                upper=1000,
                                default=10,
                                description='number of nunique values'
                            )
    min_categorical_nunique = hyperparams.Union(d, default='fraction',
                                                description='')

    agg_primitive_options = ['sum', 'std', 'max', 'skew',
                             'min', 'mean', 'count',
                             'percent_true', 'n_unique', 'mode',
                             'trend', 'median']
    default_agg_prims = ['sum', 'std', 'max', 'skew',
                         'min', 'mean', 'count',
                         'percent_true', 'nunique', 'mode']

    d = OrderedDict()
    d['agg_primitives_none'] = hyperparams.Hyperparameter[None](
        default=None,
        description='')
    d['agg_primitives_defined'] = ListHyperparam(
        options=agg_primitive_options,
        default=default_agg_prims,
        max_to_remove=8,
        description='list of Aggregation Primitives to apply.'
    )
    d['agg_primitives_custom'] = GenericListHyperparam(
        default=default_agg_prims,
        description='list of Aggregation Primitives to apply.'
    )

    agg_primitives = hyperparams.Union(d, default='agg_primitives_none',
                                       description='')

    trans_primitive_options = ['day', 'year', 'month',
                               'days', 'years', 'months',
                               'weekday', 'weekend',
                               'timesince',
                               'percentile']

    d = OrderedDict()
    default_trans_prims = ['day', 'year', 'month', 'weekday']
    d['trans_primitives_defined'] = ListHyperparam(
        options=trans_primitive_options,
        max_to_remove=8,
        description='list of Transform Primitives to apply.'
    )
    d['trans_primitives_custom'] = GenericListHyperparam(
        default=default_trans_prims,
        description='list of Transform Primitives to apply.'
    )
    d['trans_primitives_none'] = hyperparams.Hyperparameter[None](
        default=None,
        description='')
    trans_primitives = hyperparams.Union(d, default='trans_primitives_none',
                                         description='')


    sample_learning_data = hyperparams.Hyperparameter[Union[None, int]](
        description="Number of elements to sample from learningData dataframe",
        default=None,
    )

    ########
    ## Encode hyperparams
    include_unknown = hyperparams.Hyperparameter[bool](
        default=True,
        description='For .produce_encoded(), add a feature encoding the unknown class'
    )

    top_n = hyperparams.UniformInt(
                                lower=1,
                                upper=1000,
                                default=10,
                                description='For .produce_encoded(), number of top values to include in each encoding'
                            )
    remove_low_information = hyperparams.Hyperparameter[bool](
        default=True,
        description='For .produce_encoded(), indicates whether to remove features with zero variance or all null values'
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
         'version': __version__,
         'id': 'c4cd2401-6a66-4ddb-9954-33d5a5b61c52',
         'installation': [{'type': metadata_module.PrimitiveInstallationType.PIP,
                           'package_uri': 'git+https://github.com/Featuretools/ta1-primitives.git@{git_commit}#egg=featuretools_ta1-{version}'.format(
                               git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                               version=__version__
                            ),
                          }]
        })

    # Output type for this needs to be specified (and should be None)
    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed,
                         docker_containers=docker_containers)
        self._random_seed = random_seed
        # All saved attributes must be prefixed with underscores
        # Can treat hyperparams as a normal dict
        self._sample_learning_data = hyperparams['sample_learning_data']
        self._max_depth = hyperparams['max_depth']
        self._normalize_categoricals_if_single_table = \
            hyperparams['normalize_categoricals_if_single_table']
        self._find_equivalent_categories = \
            hyperparams['find_equivalent_categories']
        self._min_categorical_nunique = \
            hyperparams['min_categorical_nunique']
        self._agg_primitives = hyperparams['agg_primitives']
        self._trans_primitives = hyperparams['trans_primitives']
        self._include_unknown = hyperparams['include_unknown']
        self._top_n = hyperparams['top_n']
        self._remove_low_information = hyperparams['remove_low_information']

        # Initialize all the attributes you will eventually save
        self._target_entity = None
        self._target = None
        self._entityset = None
        self._features = None
        self._fitted = False
        self._entities_normalized = None

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
                target['column_name'],
                entities_to_normalize=entities_to_normalize,
                original_entityset=original_entityset,
                normalize_categoricals_if_single_table=self._normalize_categoricals_if_single_table,
                find_equivalent_categories=self._find_equivalent_categories,
                min_categorical_nunique=self._min_categorical_nunique,
                sample_learning_data=self._sample_learning_data)
        return entityset, target_entity, target, entities_normalized, instance_ids

    # Output type for this needs to be specified (and should be Params)
    def get_params(self) -> Params:
        features = self._features
        if features is not None:
            features = serialize_features(features)
        return Params(
            features=features,
            entityset=self._entityset,
            fitted=self._fitted,
            target_entity=self._target_entity,
            target=self._target,
            entities_normalized=cloudpickle.dumps(self._entities_normalized),
        )

    # Output type for this needs to be specified (and should be None)
    def set_params(self, *, params: Params) -> None:
        self._entityset = params['entityset']
        self._fitted = params['fitted']
        self._target_entity = params['target_entity']
        self._target = params['target']
        self._entities_normalized = cloudpickle.loads(params['entities_normalized'])
        if params['features'] is not None:
            self._features = load_features(params['features'], self._entityset)

    def __getstate__(self):
        return {'params': self.get_params(),
                'hyperparams': self.hyperparams,
                'random_seed': self.random_seed}

    def __setstate__(self, d):
        super().__init__(hyperparams=d['hyperparams'],
                         random_seed=d['random_seed'],
                         docker_containers=None)
        self.set_params(params=d['params'])
        d = d['hyperparams']
        self._sample_learning_data = d['sample_learning_data']
        self._max_depth = d['max_depth']
        self._normalize_categoricals_if_single_table = \
            d['normalize_categoricals_if_single_table']
        self._find_equivalent_categories = \
            d['find_equivalent_categories']
        self._min_categorical_nunique = \
            d['min_categorical_nunique']
        self._agg_primitives = d['agg_primitives']
        self._trans_primitives = d['trans_primitives']
        self._include_unknown = d['include_unknown']
        self._top_n = d['top_n']
        self._remove_low_information = d['remove_low_information']

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

    def produce_encoded(self, *, inputs: Input, timeout: float=None, iterations: int=None) -> CallResult[EncodedOutput]:
        feature_matrix = self.produce(inputs=inputs).value

        encoded_fm, encoded_fl = ft.encode_features(
            feature_matrix, self._features,
            top_n=self._top_n,
            include_unknown=self._include_unknown)
        if self._remove_low_information:
            encoded_fm, encoded_fl = remove_low_information_features(
                encoded_fm, encoded_fl)
        return CallResult([encoded_fm, encoded_fl])
