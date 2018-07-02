from __future__ import (absolute_import,
                        division,
                        print_function,
                        unicode_literals)
from typing import Dict, Union, Optional
import typing
from d3m.container.dataset import Dataset
from collections import OrderedDict
from d3m.container.pandas import DataFrame
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
import d3m.primitive_interfaces.unsupervised_learning as unsup
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from itertools import combinations, chain
import cloudpickle
import os
from collections import defaultdict

import featuretools as ft
from featuretools import variable_types as vtypes
from featuretools.selection import remove_low_information_features
from .utils import (D3MMetadataTypes, load_timeseries_as_df,
                    convert_variable_type)
from .normalization import normalize_categoricals
import pandas as pd
from . import __version__

ALL_ELEMENTS = metadata_base.ALL_ELEMENTS

Input = Dataset
# Featurized dataframe, indexed by the same index as Input.
# Features (columns) have human-readable names
Output = DataFrame


class Params(params.Params):
    # A named tuple for parameters.
    entityset: ft.EntitySet
    fitted: bool
    target_entity: str
    target: str
    entities_normalized: bytes
    # List of featuretools.PrimitiveBase objects representing the features.
    features: Optional[bytes]


# This is a custom hyperparameter type.
# You may find it useful if you have a list/set of
# options, and the user can select any number of them
class ListHyperparam(hyperparams.Enumeration[typing.List[str]]):
    def __init__(self, options, default=None, description=None,
                 max_to_remove=3):
        lower_limit = len(options) - max_to_remove
        upper_limit = len(options) + 1
        if default is None:
            default = sorted(options)
        else:
            default = sorted(default)

        lists = list(chain(*[list([sorted(o)
                                   for o in combinations(options, i)])
                             for i in range(lower_limit, upper_limit)]))
        super().__init__(values=lists, default=default,
                         description=description)


T = typing.TypeVar('FTPrimitive')


class GenericListHyperparam(hyperparams.Hyperparameter):
    def __init__(self, default, description=None):
        self.structural_type = typing.List[T]
        super().__init__(default=default,
                         description=description)


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
    max_depth = hyperparams.Union(
        d, default='specified',
        description='maximum allowed depth of features',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    normalize_categoricals_if_single_table = hyperparams.Hyperparameter[bool](
        default=True,
        description='''
If dataset only has a single table and
normalize_categoricals_if_single_table is True,
then normalize categoricals into separate entities.''',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    find_equivalent_categories = hyperparams.Hyperparameter[bool](
        default=True,
        description='',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
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
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                                description='')

    agg_primitive_options = ['sum', 'std', 'max', 'skew',
                             'min', 'mean', 'count',
                             'percent_true', 'n_unique', 'mode',
                             'trend', 'median']
    default_agg_prims = ['sum', 'std', 'max', 'skew',
                         'min', 'mean', 'count',
                         'percent_true', 'n_unique', 'mode']

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
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
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
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                                         description='')

    sample_learning_data = hyperparams.Hyperparameter[Union[int, None]](
        description="Number of elements to sample from learningData dataframe",
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
    # Encoder hyperparameters

    encode = hyperparams.Hyperparameter[bool](
        default=True,
        description='If True, apply One-Hot-Encoding to result',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

    include_unknown = hyperparams.Hyperparameter[bool](
        default=True,
        description='If encode is True, add a feature encoding the unknown class',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

    top_n = hyperparams.UniformInt(
        lower=1,
        upper=1000,
        default=10,
        description='If encode is True, number of top values to include in each encoding',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])

    remove_low_information = hyperparams.Hyperparameter[bool](
        default=True,
        description='Indicates whether to remove features with zero variance or all null values',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])


base_class = unsup.UnsupervisedLearnerPrimitiveBase[Input,
                                                    Output,
                                                    Params,
                                                    Hyperparams]


class DFS(base_class):
    """
    Primitive wrapping featuretools on single table datasets
    """
    __author__ = 'Feature Labs D3M team (Ben Schreck <ben.schreck@featurelabs.com>)'

    # For a list of options for each of these fields, see
    # https://metadata.datadrivendiscovery.org/
    metadata = metadata_base.PrimitiveMetadata(
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
         'installation': [{'type': metadata_base.PrimitiveInstallationType.PIP,
                           'package_uri': 'git+https://github.com/Featuretools/ta1-primitives.git@{git_commit}#egg=featuretools_ta1-{version}'.format(
                               git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                               version=__version__
                            ),
                           }
                          ]
        }
    )

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

        self._encode = hyperparams['encode']
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
        self._entityset = parsed['entityset']
        self._target_entity = parsed['target_entity']
        self._target = parsed['target']
        self._entities_normalized = parsed['entities_normalized']
        self._fitted = False

    @classmethod
    def _get_target_columns(
        # TODO: what to do if target in metadata is wrong?
        # or absent
        cls, metadata: metadata_base.DataMetadata
            ) -> typing.Sequence[metadata_base.SimpleSelectorSegment]:

        target_columns = []
        n_resources = metadata.query(())['dimension']['length']
        for res_id in range(n_resources):
            stypes = metadata.query((str(res_id), ))['semantic_types']
            if D3MMetadataTypes.EntryPoint in stypes:
                # is learning data resource
                ncolumns = metadata.query((str(res_id), ALL_ELEMENTS))['dimension']['length']
                for column_index in range(ncolumns):
                    column_metadata = metadata.query((str(res_id), ALL_ELEMENTS,
                                                      column_index))
                    semantic_types = column_metadata.get('semantic_types', [])
                    if D3MMetadataTypes.TrueTarget in semantic_types:
                        column_name = column_metadata['name']
                        target_columns.append(column_name)
        return target_columns

    @classmethod
    def _get_target_column(
        cls, metadata: metadata_base.DataMetadata
            ) -> typing.Sequence[metadata_base.SimpleSelectorSegment]:
        targets = cls._get_target_columns(metadata=metadata)
        if len(targets):
            return targets[0]
        raise ValueError("No targets specified in metadata")

    def _parse_inputs(self, inputs, entities_to_normalize=None,
                      original_entityset=None):
        target = self._get_target_column(inputs.metadata)

        parsed = self._convert_d3m_dataset_to_entityset(
                inputs,
                target,
                entities_to_normalize=entities_to_normalize,
                original_entityset=original_entityset,
                normalize_categoricals_if_single_table=self._normalize_categoricals_if_single_table,
                find_equivalent_categories=self._find_equivalent_categories,
                min_categorical_nunique=self._min_categorical_nunique,
                sample_learning_data=self._sample_learning_data)
        parsed['target'] = target
        return parsed

    # Output type for this needs to be specified (and should be Params)
    def get_params(self) -> Params:
        return Params(
            features=cloudpickle.dumps(self._features),
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
        self._entities_normalized = cloudpickle.loads(
            params['entities_normalized'])
        self._features = cloudpickle.loads(params['features'])

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

    @classmethod
    def _update_metadata(
            cls,
            metadata: metadata_base.DataMetadata,
            resource_id: str,
            target: str,
            features: list,
            for_value: DataFrame = None,
            source: typing.Any = None) -> metadata_base.DataMetadata:
        if source is None:
            source = cls

        resource_metadata = dict(metadata.query((resource_id,)))

        resource_metadata.update(
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': DataFrame,
                'target_column': target,
            },
        )

        new_metadata = metadata.clear(resource_metadata,
                                      for_value=for_value,
                                      source=source)

        new_metadata = cls._copy_elements_metadata(
            metadata, (resource_id,), (), new_metadata, source=source)

        old_resource_metadata = metadata.query((resource_id, ALL_ELEMENTS))
        resource_metadata = dict(old_resource_metadata)
        resource_metadata['dimension'] = {'length': len(features)}
        resource_metadata['ft_features'] = cloudpickle.dumps(features)
        new_metadata = new_metadata.update((ALL_ELEMENTS,), resource_metadata)

        for i, f in enumerate(features):
            resource_metadata = dict(new_metadata.query((ALL_ELEMENTS, i)))
            resource_metadata['semantic_types'] = [D3MMetadataTypes.to_d3m(
                f.variable_type)]
            resource_metadata['name'] = f.get_name()
            # TODO: structural type
            new_metadata = new_metadata.update((ALL_ELEMENTS, i),
                                               resource_metadata)
        return new_metadata

    def _convert_d3m_dataset_to_entityset(
            self, ds, target_colname,
            entities_to_normalize=None,
            original_entityset=None,
            normalize_categoricals_if_single_table=True,
            find_equivalent_categories=True,
            min_categorical_nunique=0.1,
            sample_learning_data=None):
        n_resources = ds.metadata.query(())['dimension']['length']
        tables = {}
        keys = defaultdict(dict)
        entityset = ft.EntitySet(ds.metadata.query(())['id'])
        instance_ids = None
        learning_data_res_id = None
        for i in range(n_resources):
            variable_types = {}
            index = None
            time_index = None
            res_id = str(i)
            stypes = ds.metadata.query((res_id,))['semantic_types']
            res = ds[res_id]
            if D3MMetadataTypes.Table in stypes:
                tables[res_id] = res
            elif D3MMetadataTypes.Timeseries in stypes:
                df, index, time_index = load_timeseries_as_df(ds, res_id)
            else:
                continue
            if D3MMetadataTypes.EntryPoint in stypes:
                learning_data_res_id = res_id
                assert 'd3mIndex' in res,\
                    "Could not find d3mIndex in learningData table"

            for icol, col in enumerate(res.columns):
                col_metadata = ds.metadata.query((res_id, ALL_ELEMENTS, icol))
                col_stypes = col_metadata['semantic_types']

                # TODO: figure out a way to tell if column
                # is justa refernce to filename in other table
                if D3MMetadataTypes.Privileged in col_stypes:
                    del res[col]
                elif 'foreign_key' in col_metadata:
                    assert col_metadata['foreign_key']['type'] == 'COLUMN',\
                        "Foreign key resource to non-tabular entry"
                    keys[res_id][col] = col_metadata['foreign_key']
                    vtype = vtypes.Id
                elif D3MMetadataTypes.PrimaryKey in col_stypes:
                    index = col
                    vtype = vtypes.Index
                else:
                    column_mtype = [st for st in col_stypes
                                    if st not in D3MMetadataTypes.KeyTypes
                                    and D3MMetadataTypes.is_column_type(st)][0]
                    vtype = D3MMetadataTypes.to_ft(column_mtype)
                    # TODO: look into why timeIndicator and
                    # dateTime are the same
                    time_types = (D3MMetadataTypes.TimeIndicator,
                                  D3MMetadataTypes.Datetime)
                    tried_to_make_time_index = False
                    if time_index is None and column_mtype in time_types:
                        time_index = col
                        vtype = vtypes.DatetimeTimeIndex
                        tried_to_make_time_index = True
                    vtype = convert_variable_type(res, col, vtype,
                                                  target_colname)
                    if tried_to_make_time_index:
                        if vtype == vtypes.Numeric:
                            vtype = vtypes.NumericTimeIndex
                        elif vtype != vtypes.DatetimeTimeIndex:
                            time_index = None
                variable_types[col] = vtype
            if res_id == learning_data_res_id:
                res['d3mIndex'] = res['d3mIndex'].astype(int)

                if original_entityset is not None:
                    original_learning_data = original_entityset[res_id].df
                    res = (pd.concat([res, original_learning_data])
                             .drop_duplicates(['d3mIndex']))
                if sample_learning_data:
                    res = res.sample(sample_learning_data)
                instance_ids = res['d3mIndex']

            make_index = False
            if not index:
                index = "res-{}-id".format(res_id)
                make_index = True
            entityset.entity_from_dataframe(res_id,
                                            res,
                                            index=index,
                                            make_index=make_index,
                                            time_index=time_index,
                                            variable_types=variable_types)

        entities_normalized = None
        if normalize_categoricals_if_single_table and len(tables) == 1:
            entities_normalized = normalize_categoricals(
                    entityset,
                    res_id,
                    ignore_columns=[target_colname],
                    entities_to_normalize=entities_to_normalize,
                    find_equivalent_categories=find_equivalent_categories,
                    min_categorical_nunique=min_categorical_nunique)
        else:
            for res_id, _keys in keys.items():
                for col_name, fkey_info in _keys.items():
                    foreign_res_id = fkey_info['resource_id']
                    foreign_col = fkey_info['column_name']
                    ft_var = entityset[res_id][col_name]
                    ft_foreign_var = entityset[foreign_res_id][foreign_col]
                    entityset.add_relationship(
                        ft.Relationship(ft_foreign_var, ft_var))

        return {
            'entityset': entityset,
            'target_entity': learning_data_res_id,
            'entities_normalized': entities_normalized,
            'instance_ids': instance_ids,
        }

    @classmethod
    def _copy_elements_metadata(
            cls,
            source_metadata: metadata_base.DataMetadata,
            selector_prefix: metadata_base.Selector,
            selector: metadata_base.Selector,
            target_metadata: metadata_base.DataMetadata,
            *, source: typing.Any = None) -> metadata_base.DataMetadata:
        '''Taken from common_primitives.data_to_dataframe
        '''
        if source is None:
            source = cls

        elements = source_metadata.get_elements(
            list(selector_prefix) + list(selector))

        for element in elements:
            new_selector = list(selector) + [element]
            metadata = source_metadata.query(
                list(selector_prefix) + new_selector)
            target_metadata = target_metadata.update(
                new_selector, metadata, source=source)
            target_metadata = cls._copy_elements_metadata(
                source_metadata, selector_prefix, new_selector,
                target_metadata, source=source)

        return target_metadata

    @classmethod
    def can_accept(
            cls, *,
            method_name: str,
            arguments: typing.Dict[str,
                                   typing.Union[metadata_base.Metadata, type]]
            ) -> typing.Optional[metadata_base.DataMetadata]:
        output_metadata = super().can_accept(
            method_name=method_name, arguments=arguments)

        # If structural types didn't match, don't bother.
        if output_metadata is None:
            return None

        if 'inputs' not in arguments:
            return output_metadata

        inputs_metadata = typing.cast(metadata_base.DataMetadata,
                                      arguments['inputs'])

        # TODO: check if each resource is tabular or timeseries
        # for element in inputs_metadata.get_elements(()):
        #     if element is metadata_base.ALL_ELEMENTS:
        #         continue

        #     resource_id = typing.cast(str, element)

        target_columns = cls._get_target_columns(inputs_metadata)
        if not target_columns:
            raise ValueError("Input data has no target columns.")
        if len(target_columns) > 1:
            # TODO: add this check to can_accept
            raise ValueError("Can only accept datasets with a single target")

        return output_metadata
        # TODO: we don't want to updat eth emetadata
        # before we know what features will be generated right?
        # or do we want to run DFS features-only here?
        # return cls._update_metadata(inputs_metadata, source=cls)

    def _fit_and_return_result(self, *,
                               timeout: float=None,
                               iterations: int=None):
        if self._entityset is None:
            raise ValueError("Must call .set_training_data() ",
                             "before calling .fit()")
        ignore_variables = {self._target_entity: [self._target]}
        time_index = self._entityset[self._target_entity].time_index
        index = self._entityset[self._target_entity].index
        cutoff_time = None
        if time_index:
            target_df = self._entityset[self._target_entity].df
            cutoff_time = target_df[[index, time_index]]
            ignore_variables = None

        features_only = not self._encode and not self._remove_low_information

        res = ft.dfs(entityset=self._entityset,
                     target_entity=self._target_entity,
                     cutoff_time=cutoff_time,
                     features_only=features_only,
                     ignore_variables=ignore_variables,
                     max_depth=self._max_depth,
                     agg_primitives=self._agg_primitives,
                     trans_primitives=self._trans_primitives)
        if not features_only:
            if self._encode:
                fm, self._features = ft.encode_features(
                    *res, top_n=self._top_n,
                    include_unknown=self._include_unknown)
            if self._remove_low_information:
                fm, self._features = remove_low_information_features(
                    fm, self._features)
            self._fitted = True
            return fm
        else:
            self._fitted = True
            self._features = res

    def produce(self, *, inputs: Input,
                timeout: float=None,
                iterations: int=None) -> CallResult[Output]:
        if self._features is None:
            raise ValueError("Must call fit() before calling produce()")
        if not isinstance(inputs, Dataset):
            raise ValueError("Inputs to produce() must be a Dataset")
        features = self._features

        parsed = self._parse_inputs(
            inputs,
            entities_to_normalize=self._entities_normalized,
            original_entityset=self._entityset)
        entityset = parsed['entityset']
        target = parsed['target']
        instance_ids = parsed['instance_ids']

        feature_matrix = ft.calculate_feature_matrix(features,
                                                     entityset=entityset,
                                                     instance_ids=instance_ids,
                                                     cutoff_time_in_index=True)

        feature_matrix = (feature_matrix.reset_index('time')
                                        .loc[instance_ids, :]
                                        .set_index('time', append=True))
        for f in features:
            if issubclass(f.variable_type, vtypes.Discrete):
                as_obj = feature_matrix[f.get_name()].astype(object)
                feature_matrix[f.get_name()] = as_obj
            elif issubclass(f.variable_type, vtypes.Numeric):
                as_num = pd.to_numeric(feature_matrix[f.get_name()])
                feature_matrix[f.get_name()] = as_num
            elif issubclass(f.variable_type, vtypes.Datetime):
                as_date = pd.to_datetime(feature_matrix[f.get_name()])
                feature_matrix[f.get_name()] = as_date

        fm_with_metadata = DataFrame(feature_matrix)

        fm_with_metadata.metadata = self._update_metadata(
            inputs.metadata,
            resource_id=self._target_entity,
            features=self._features,
            target=target,
            for_value=fm_with_metadata,
            source=self)
        return CallResult(fm_with_metadata)

    def fit(self, *,
            timeout: float=None,
            iterations: int=None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)
        self._fit_and_return_result(timeout=timeout,
                                    iterations=iterations)
        return CallResult(None)

    def fit_produce(self, *, inputs: Input,
                    timeout: float=None,
                    iterations: int=None) -> CallResult[Output]:
        self.set_training_data(inputs=inputs)
        fm = self._fit_and_return_result(timeout=timeout,
                                         iterations=iterations)
        if fm is None:
            fm = self.produce(inputs=inputs,
                              timeout=timeout,
                              iterations=iterations).value
        return CallResult(fm)
