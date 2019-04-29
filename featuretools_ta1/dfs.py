import os
import typing
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Union

import cloudpickle
import featuretools as ft
import numpy as np
import pandas as pd
from d3m import index, utils
from d3m.container.dataset import Dataset
from d3m.container.pandas import DataFrame
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams, params
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from featuretools import variable_types
from featuretools.selection import remove_low_information_features

from featuretools_ta1 import __version__
from featuretools_ta1.normalization import normalize_categoricals
from featuretools_ta1.utils import (
    D3MMetadataTypes, convert_variable_type, fast_concat, get_target_columns,
    load_timeseries_as_df)

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


DEFAULT_PRIMITIVES = [
    'characters'
    'count',
    'day',
    'max',
    'mean',
    'min',
    'mode'
    'month',
    'num_unique',
    'numwords',
    'percent_true',
    'skew',
    'std',
    'sum',
    'weekday',
    'year',
]
ALL_PRIMITIVES = [
    # 'absolute',
    'avg_time_between',
    'characters',
    'count',
    # 'day',
    'hour',
    'is_weekend',
    # 'last',
    # 'latitude',
    'max',
    'mean',
    'median',
    'min',
    # 'minute',
    'mode',
    # 'month',
    # 'n_most_common',
    # 'negate',
    'num_true',
    'num_unique',
    'numwords',
    'percent_true',
    # 'percentile',
    # 'second',
    'skew',
    'std',
    'sum',
    # 'time_since_last',
    'week',
    'weekday',
    'year',
]


def _get_primitive_hyperparams():
    """Get one boolean hyperparam object for each available featuretools primitive.

    The hyperparameter will be named {primitimve_type}_{primitive_name},
    will have the primitive description, and will default to True or False
    depending on whether the primitive name is in the _DEAFULT_PRIMITIVES list.

    An example of such a primitive is::

        aggregation_max = hyperparams.Hyperparameter[bool](
            description='Finds the maximum non-null value of a numeric feature.',
            default=True,
            semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
        )

    Returns:
        dict containing hyperparameter names as keys and hyperparameter objects as values.
    """
    primitive_hyperparams = dict()
    primitives = ft.list_primitives()
    for _, primitive in primitives.iterrows():
        primitive_name = primitive['name']
        if primitive_name in ALL_PRIMITIVES:
            hyperparam_name = '{}_{}'.format(primitive['type'], primitive_name)
            hyperparam = hyperparams.Hyperparameter[bool](
                default=primitive_name in DEFAULT_PRIMITIVES,
                description=primitive['description'],
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
            )
            primitive_hyperparams[hyperparam_name] = hyperparam

    return dict(sorted(primitive_hyperparams.items()))


class Hyperparams(hyperparams.Hyperparams):

    # D3M specific
    include_target_in_output = hyperparams.Hyperparameter[bool](
        default=True,
        description='Include target column in output dataframe',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    sample_learning_data = hyperparams.Hyperparameter[Union[int, None]](
        description='Number of elements to sample from learningData dataframe',
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    # DFS arguments
    max_depth = hyperparams.UniformInt(
        lower=1,
        upper=5,
        default=2,
        description='Maximum allowed depth of features',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    # Normalization Options
    normalize_single_table = hyperparams.Hyperparameter[bool](
        default=True,
        description=(
            'If dataset has only one table and normalize_single_table '
            'is True, normalize categoricals into separate entities.'
        ),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    find_equivalent_categories = hyperparams.Hyperparameter[bool](
        default=True,
        description='',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    min_categorical_nunique = hyperparams.Union(
        OrderedDict([
            ('fraction', hyperparams.Uniform(
                lower=0.00001,
                upper=1,
                default=.1,
                description='fraction of nunique values'
            )),
            ('value', hyperparams.UniformInt(
                lower=1,
                upper=1000,
                default=10,
                description='number of nunique values'
            ))
        ]),
        default='fraction',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description=''
    )

    # Encoding hyperparameters
    encode = hyperparams.Hyperparameter[bool](
        default=True,
        description='If True, apply One-Hot-Encoding to result',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    include_unknown = hyperparams.Hyperparameter[bool](
        default=True,
        description='If encode is True, add a feature encoding the unknown class',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    top_n = hyperparams.UniformInt(
        lower=1,
        upper=1000,
        default=10,
        description='If encode is True, number of top values to include in each encoding',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    remove_low_information = hyperparams.Hyperparameter[bool](
        default=True,
        description='Indicates whether to remove features with zero variance or all null values',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    # Primitive Hyperparameters
    locals().update(_get_primitive_hyperparams())


class DFS(UnsupervisedLearnerPrimitiveBase[Input, Output, Params, Hyperparams]):
    """Primitive wrapping featuretools on single table datasets."""

    __author__ = 'Feature Labs D3M team (Max Kanter <max.kanter@featurelabs.com>)'

    # For a list of options for each of these fields, see
    # https://metadata.datadrivendiscovery.org/
    _git_commit = utils.current_git_commit(os.path.dirname(__file__))
    _package_uri = (
        'git+https://github.com/Featuretools/ta1-primitives.git'
        '@{git_commit}#egg=featuretools_ta1'
    ).format(git_commit=_git_commit, version=__version__)

    _python_path = 'd3m.primitives.feature_construction.deep_feature_synthesis.Featuretools'

    metadata = metadata_base.PrimitiveMetadata(
        {
            'algorithm_types': ['DEEP_FEATURE_SYNTHESIS'],
            'name': 'Deep Feature Synthesis',
            'primitive_family': 'FEATURE_CONSTRUCTION',
            'python_path': _python_path,
            'source': {
                'name': 'MIT_FeatureLabs',
                'contact': 'mailto://max.kanter@featurelabs.com',
                'uris': ['https://doc.featuretools.com'],
                'license': 'BSD-3-Clause'
            },
            'description': (
                'Calculates a feature matrix and features given a '
                'single-table tabular D3M Dataset.'
            ),
            'keywords': [
                'featurization',
                'feature engineering',
                'feature extraction',
                'feature construction'
            ],
            'hyperparameters_to_tune': ['max_depth', 'normalize_single_table'],
            'version': __version__,
            'id': 'c4cd2401-6a66-4ddb-9954-33d5a5b61c52',
            'installation': [
                {
                    'type': metadata_base.PrimitiveInstallationType.PIP,
                    'package_uri': _package_uri
                }
            ]
        }
    )

    # TODO: look into why timeIndicator and
    # dateTime are the same
    _TIME_TYPES = (D3MMetadataTypes.TimeIndicator, D3MMetadataTypes.Datetime)

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed,
                         docker_containers=docker_containers)

        # Initialize all the attributes you will eventually save
        self._target_entity = None
        self._target = None
        self._entityset = None
        self._features = None
        self._fitted = False
        self._entities_normalized = None

    # Output type for this needs to be specified (and should be None)
    def set_training_data(self, *, inputs: Input) -> None:
        parsed = self._parse_inputs(inputs, parse_target=True)
        self._entityset = parsed['entityset']
        self._target_entity = parsed['target_entity']
        self._target = parsed['target']
        self._entities_normalized = parsed['entities_normalized']
        self._fitted = False

    @classmethod
    def _get_target_column(
        cls, metadata: metadata_base.DataMetadata
    ) -> typing.Sequence[metadata_base.SimpleSelectorSegment]:

        targets = get_target_columns(metadata=metadata)
        if len(targets):
            return targets[0]

        raise ValueError('No targets specified in metadata')

    def _parse_inputs(self, inputs, entities_to_normalize=None,
                      original_entityset=None, parse_target=False):

        target = self._target
        if self._target is None or parse_target:
            target = self._get_target_column(inputs.metadata)

        parsed = self._convert_d3m_dataset_to_entityset(
            inputs,
            target,
            entities_to_normalize=entities_to_normalize,
            original_entityset=original_entityset
        )

        if self._target is None:
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
        self._features = cloudpickle.loads(params['features'])
        self._entityset = params['entityset']
        self._fitted = params['fitted']
        self._target_entity = params['target_entity']
        self._target = params['target']
        self._entities_normalized = cloudpickle.loads(params['entities_normalized'])

    def __getstate__(self):
        return {
            'params': self.get_params(),
            'hyperparams': self.hyperparams,
            'random_seed': self.random_seed
        }

    def __setstate__(self, state):
        super().__init__(
            hyperparams=state['hyperparams'],
            random_seed=state['random_seed'],
            docker_containers=None
        )
        self.set_params(params=state['params'])

    @classmethod
    def _update_metadata(
            cls,
            metadata: metadata_base.DataMetadata,
            resource_id: str,
            target: str,
            features: list,
            update_target: bool = True,
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

        new_metadata = metadata.clear(resource_metadata, for_value=for_value, source=source)

        new_metadata = cls._copy_elements_metadata(
            metadata, (resource_id,), (), new_metadata, source=source)

        old_resource_metadata = metadata.query((resource_id, ALL_ELEMENTS))
        resource_metadata = dict(old_resource_metadata)
        # TODO: what if we don't include target? len(features) + 1
        resource_metadata['dimension'] = {'length': len(features) + 1}
        if update_target:
            resource_metadata['dimension'] = {'length': len(features) + 2}

        resource_metadata['ft_features'] = cloudpickle.dumps(features)
        new_metadata = new_metadata.update((ALL_ELEMENTS,), resource_metadata)

        for i, f in enumerate(features):
            resource_metadata = {
                'semantic_types': [D3MMetadataTypes.to_d3m(f.variable_type),
                                   D3MMetadataTypes.Attribute],
                'structural_type': D3MMetadataTypes.to_default_structural_type(f.variable_type),
                'name': f.get_name()
            }
            if for_value is not None:
                structural_type = type(for_value[f.get_name()].iloc[0])
                resource_metadata['structural_type'] = structural_type

            new_metadata = new_metadata.update((ALL_ELEMENTS, i),
                                               resource_metadata)

        # update index
        resource_metadata = {'semantic_types': [D3MMetadataTypes.PrimaryKey],
                             'name': 'd3mIndex'}

        if for_value is not None:
            structural_type = type(for_value['d3mIndex'].iloc[0])
            resource_metadata['structural_type'] = structural_type

        new_metadata = new_metadata.update((ALL_ELEMENTS, len(features)),
                                           resource_metadata)
        if update_target:
            resource_metadata = {'semantic_types': [D3MMetadataTypes.TrueTarget,
                                                    D3MMetadataTypes.SuggestedTarget],
                                 'name': target}

            if for_value is not None:
                structural_type = type(for_value[target].iloc[0])
                resource_metadata['structural_type'] = structural_type

            new_metadata = new_metadata.update((ALL_ELEMENTS, len(features) + 1),
                                               resource_metadata)

        return new_metadata

    def _convert_d3m_dataset_to_entityset(self, inputs, target,
                                          entities_to_normalize=None, original_entityset=None):

        tables = {}
        keys = defaultdict(dict)
        entityset = ft.EntitySet(inputs.metadata.query(())['id'])
        instance_ids = None
        learning_data_res_id = None

        for res_id in inputs.metadata.get_elements(()):
            variables = {}
            index = None
            time_index = None
            stypes = inputs.metadata.query((res_id,))['semantic_types']
            res = inputs[res_id]
            if D3MMetadataTypes.Table in stypes:
                tables[res_id] = res
            elif D3MMetadataTypes.Timeseries in stypes:
                df, index, time_index = load_timeseries_as_df(inputs, res_id)
            else:
                continue

            if D3MMetadataTypes.EntryPoint in stypes:
                learning_data_res_id = res_id
                assert 'd3mIndex' in res, 'Could not find d3mIndex in learningData table'

            for icol, col in enumerate(res.columns):
                col_metadata = inputs.metadata.query((res_id, ALL_ELEMENTS, icol))
                col_stypes = col_metadata['semantic_types']

                # TODO: figure out a way to tell if column
                # is justa refernce to filename in other table
                if D3MMetadataTypes.Privileged in col_stypes:
                    del res[col]

                elif 'foreign_key' in col_metadata:
                    type_column = col_metadata['foreign_key']['type'] == 'COLUMN'
                    assert type_column, 'Foreign key resource to non-tabular entry'

                    keys[res_id][col] = col_metadata['foreign_key']
                    vtype = variable_types.Id

                elif D3MMetadataTypes.PrimaryKey in col_stypes:
                    index = col
                    vtype = variable_types.Index

                else:
                    for st in col_stypes:
                        not_in_keytypes = st not in D3MMetadataTypes.KeyTypes
                        if not_in_keytypes and D3MMetadataTypes.is_column_type(st):
                            column_mtype = st
                            break

                    vtype = D3MMetadataTypes.to_ft(column_mtype)

                    tried_to_make_time_index = False
                    if time_index is None and column_mtype in self._TIME_TYPES:
                        time_index = col
                        vtype = variable_types.DatetimeTimeIndex
                        tried_to_make_time_index = True

                    vtype = convert_variable_type(res, col, vtype, target)

                    if tried_to_make_time_index:
                        if vtype == variable_types.Numeric:
                            vtype = variable_types.NumericTimeIndex

                        elif vtype != variable_types.DatetimeTimeIndex:
                            time_index = None

                variables[col] = vtype

            if res_id == learning_data_res_id:
                res['d3mIndex'] = res['d3mIndex'].astype(int)

                if original_entityset is not None:
                    original_learning_data = original_entityset[res_id].df
                    res = (fast_concat([res, original_learning_data])
                           .drop_duplicates(['d3mIndex']))

                if self.hyperparams['sample_learning_data']:
                    res = res.sample(self.hyperparams['sample_learning_data'])

                instance_ids = res['d3mIndex']

            make_index = False
            if not index:
                index = 'res-{}-id'.format(res_id)
                make_index = True

            entityset.entity_from_dataframe(
                res_id,
                res,
                index=index,
                make_index=make_index,
                time_index=time_index,
                variable_types=variables
            )

        entities_normalized = None
        if self.hyperparams['normalize_single_table'] and len(tables) == 1:
            entities_normalized = normalize_categoricals(
                entityset,
                res_id,
                ignore_columns=[target],
                entities_to_normalize=entities_to_normalize,
                find_equivalent_categories=self.hyperparams['find_equivalent_categories'],
                min_categorical_nunique=self.hyperparams['min_categorical_nunique']
            )

        else:
            for res_id, _keys in keys.items():
                for col_name, fkey_info in _keys.items():
                    foreign_res_id = fkey_info['resource_id']
                    foreign_col = fkey_info['column_name']
                    ft_var = entityset[res_id][col_name]
                    ft_foreign_var = entityset[foreign_res_id][foreign_col]
                    entityset.add_relationship(ft.Relationship(ft_foreign_var, ft_var))

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

        if source is None:
            source = cls

        elements = source_metadata.get_elements(list(selector_prefix) + list(selector))

        for element in elements:
            new_selector = list(selector) + [element]
            metadata = source_metadata.query(list(selector_prefix) + new_selector)

            target_metadata = target_metadata.update(new_selector, metadata, source=source)
            target_metadata = cls._copy_elements_metadata(
                source_metadata, selector_prefix, new_selector,
                target_metadata, source=source)

        return target_metadata

    @classmethod
    def can_accept(
        cls, *,
        method_name: str,
        arguments: typing.Dict[str, typing.Union[metadata_base.Metadata, type]]
    ) -> typing.Optional[metadata_base.DataMetadata]:

        output_metadata = super().can_accept(
            method_name=method_name, arguments=arguments)

        # If structural types didn't match, don't bother.
        if output_metadata is None:
            return None

        if 'inputs' not in arguments:
            return output_metadata

        inputs_metadata = typing.cast(metadata_base.DataMetadata, arguments['inputs'])

        target_columns = cls._get_target_columns(inputs_metadata)
        if not target_columns:
            raise ValueError('Input data has no target columns.')

        if len(target_columns) > 1:
            # TODO: add this check to can_accept
            raise ValueError('Can only accept datasets with a single target')

        return output_metadata

    def _fit_and_return_result(self, *, timeout: float = None, iterations: int = None):

        if self._entityset is None:
            raise ValueError('Must call .set_training_data() before calling .fit()')

        ignore_variables = {self._target_entity: [self._target]}
        time_index = self._entityset[self._target_entity].time_index
        index = self._entityset[self._target_entity].index
        cutoff_time = None
        if time_index:
            target_df = self._entityset[self._target_entity].df
            cutoff_time = target_df[[index, time_index]]
            ignore_variables = None

        features_only = (
            not self.hyperparams['encode'] and not self.hyperparams['remove_low_information']
        )

        agg_primitives = [
            name[12:]
            for name, value in self.hyperparams.items()
            if name.startswith('aggregation_') and value
        ]
        trans_primitives = [
            name[10:]
            for name, value in self.hyperparams.items()
            if name.startswith('transform_') and value
        ]

        res = ft.dfs(
            entityset=self._entityset,
            target_entity=self._target_entity,
            cutoff_time=cutoff_time,
            cutoff_time_in_index=False,
            features_only=features_only,
            ignore_variables=ignore_variables,
            max_depth=self.hyperparams['max_depth'],
            agg_primitives=agg_primitives,
            trans_primitives=trans_primitives
        )

        if not features_only:
            if self.hyperparams['encode']:
                fm, self._features = ft.encode_features(
                    *res, top_n=self.hyperparams['top_n'],
                    include_unknown=self.hyperparams['include_unknown'])

            if self.hyperparams['remove_low_information']:
                fm, self._features = remove_low_information_features(fm, self._features)

            self._fitted = True

            return fm

        else:
            self._fitted = True
            self._features = res

    def _format_fm_after_cfm(self, feature_matrix, instance_ids, features,
                             target, entityset, inputs_metadata):

        feature_matrix = feature_matrix.loc[instance_ids, :]
        for f in features:
            if issubclass(f.variable_type, variable_types.Discrete):
                as_obj = feature_matrix[f.get_name()].astype(object)
                feature_matrix[f.get_name()] = as_obj

            elif issubclass(f.variable_type, variable_types.Numeric):
                as_num = pd.to_numeric(feature_matrix[f.get_name()])
                feature_matrix[f.get_name()] = as_num

            elif issubclass(f.variable_type, variable_types.Datetime):
                as_date = pd.to_datetime(feature_matrix[f.get_name()])
                feature_matrix[f.get_name()] = as_date

        additional_columns = ['d3mIndex']
        if target not in feature_matrix.columns and self.hyperparams['include_target_in_output']:
            if target not in entityset[self._target_entity].df:
                feature_matrix[target] = np.nan

            else:
                target_col = entityset[self._target_entity].df[target]
                target_values = target_col.loc[instance_ids].values
                if (target_values == '').all():
                    feature_matrix[target] = np.nan
                else:
                    feature_matrix[target] = target_col.loc[instance_ids].values

            additional_columns.append(target)

        feature_matrix.reset_index(inplace=True)
        feature_matrix = feature_matrix[[f.get_name() for f in features] + additional_columns]

        fm_with_metadata = DataFrame(feature_matrix)

        fm_with_metadata.metadata = self._update_metadata(
            inputs_metadata,
            resource_id=self._target_entity,
            features=self._features,
            target=target,
            for_value=fm_with_metadata,
            update_target=self.hyperparams['include_target_in_output'],
            source=self
        )

        return fm_with_metadata

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:

        if self._fitted:
            return CallResult(None)

        self._fit_and_return_result(timeout=timeout, iterations=iterations)

        return CallResult(None)

    def produce(self, *, inputs: Input,
                timeout: float = None,
                iterations: int = None) -> CallResult[Output]:

        if self._features is None:
            raise ValueError('Must call fit() before calling produce()')

        if not isinstance(inputs, Dataset):
            raise ValueError('Inputs to produce() must be a Dataset')

        features = self._features

        parsed = self._parse_inputs(
            inputs,
            entities_to_normalize=self._entities_normalized,
            # original_entityset=self._entityset,
            parse_target=False
        )

        entityset = parsed['entityset']
        target = self._target
        instance_ids = parsed['instance_ids']

        feature_matrix = ft.calculate_feature_matrix(
            features,
            entityset=entityset,
            instance_ids=instance_ids,
            cutoff_time_in_index=False
        )

        fm_with_metadata = self._format_fm_after_cfm(
            feature_matrix,
            instance_ids,
            features,
            target,
            entityset,
            inputs.metadata
        )

        return CallResult(fm_with_metadata)

    def fit_produce(self, *, inputs: Input, timeout: float = None,
                    iterations: int = None) -> CallResult[Output]:

        self.set_training_data(inputs=inputs)
        fm = self._fit_and_return_result(timeout=timeout, iterations=iterations)

        if fm is None:
            fm = self.produce(
                inputs=inputs,
                timeout=timeout,
                iterations=iterations
            ).value
        else:
            fm = self._format_fm_after_cfm(
                fm,
                fm.index,
                self._features,
                self._target,
                self._entityset,
                inputs.metadata
            )

        return CallResult(fm)

    @classmethod
    def get_demo_pipeline(cls) -> None:

        # Creating pipeline
        pipeline = Pipeline(context=Context.TESTING)
        pipeline.add_input(name='inputs')

        # Step 0: DFS
        step_0 = PrimitiveStep(primitive=cls)
        step_0.add_argument(
            name='inputs',
            argument_type=ArgumentType.CONTAINER,
            data_reference='inputs.0'
        )
        step_0.add_output('produce')
        pipeline.add_step(step_0)

        # Step 1: SKlearnImputer
        sklearn_imputer = index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn')
        step_1 = PrimitiveStep(primitive=sklearn_imputer)
        step_1.add_argument(
            name='inputs',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.0.produce'
        )
        step_1.add_output('produce')
        pipeline.add_step(step_1)

        # Step 2: SKlearnRFC
        sklearn_rf = index.get_primitive('d3m.primitives.classification.random_forest.SKlearn')
        step_2 = PrimitiveStep(primitive=sklearn_rf)
        step_2.add_hyperparameter(
            name='use_semantic_types',
            argument_type=ArgumentType.VALUE,
            data=True
        )
        step_2.add_hyperparameter(
            name='add_index_columns',
            argument_type=ArgumentType.VALUE,
            data=True
        )
        step_2.add_argument(
            name='inputs',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.1.produce'
        )
        step_2.add_argument(
            name='outputs',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.1.produce'
        )
        step_2.add_output('produce')
        pipeline.add_step(step_2)

        # Step 3: ConstructPredictions
        df_common = index.get_primitive(
            'd3m.primitives.data_transformation.construct_predictions.DataFrameCommon')
        step_3 = PrimitiveStep(primitive=df_common)
        step_3.add_argument(
            name='inputs',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.2.produce'
        )
        step_3.add_argument(
            name='reference',
            argument_type=ArgumentType.CONTAINER,
            data_reference='steps.2.produce'
        )
        step_3.add_output('produce')
        pipeline.add_step(step_3)

        # Final Output
        pipeline.add_output(name='output predictions', data_reference='steps.3.produce')

        pipeline.dataset = '185_baseball'

        return pipeline
