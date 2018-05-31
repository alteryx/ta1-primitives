from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from typing import Dict
import typing
from d3m.container.pandas import DataFrame
from d3m.metadata import hyperparams, base as metadata_base
from d3m import utils
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult, DockerContainer
import os

import featuretools as ft
from .utils import D3MMetadataTypes
from . import __version__
import cloudpickle

ALL_ELEMENTS = metadata_base.ALL_ELEMENTS

Input = DataFrame
Output = DataFrame


class Hyperparams(hyperparams.Hyperparams):
    include_unknown = hyperparams.Hyperparameter[bool](
        default=True,
        description='Add a feature encoding the unknown class'
    )

    top_n = hyperparams.UniformInt(
        lower=1,
        upper=1000,
        default=10,
        description='Number of top values to include in each encoding'
    )
    remove_low_information = hyperparams.Hyperparameter[bool](
        default=True,
        description='''Indicates whether to remove
features with zero variance or all null values'''
    )


class Encoder(TransformerPrimitiveBase[Input, Output, Hyperparams]):
    """
    Primitive wrapping ft.encode_features
    """
    __author__ = 'Feature Labs D3M team (Ben Schreck <ben.schreck@featurelabs.com>)'

    # For a list of options for each of these fields, see
    # https://metadata.datadrivendiscovery.org/
    metadata = metadata_base.PrimitiveMetadata(
        {'algorithm_types': ['ENCODE_ONE_HOT', ],
         'name': 'One Hot Encoder',
         'primitive_family': 'FEATURE_CONSTRUCTION',
         'python_path': 'd3m.primitives.featuretools_ta1.Encoder',
         "source": {
           "name": "MIT_FeatureLabs",
           "contact": "mailto://ben.schreck@featurelabs.com",
           "uris": ["https://doc.featuretools.com"],
           "license": "BSD-3-Clause"

         },
         "description": "Calculates an encoded feature matrix based on meta-information in a DataFrame detailing which columns are Categorical.",
         "keywords": ["featurization", "feature engineering", "feature extraction"],
         "hyperparameters_to_tune": ["top_n", "remove_low_information", "include_unknown"],
         'version': __version__,
         'id': 'c4cd2401-6a66-4ddb-9954-33d5a5b61c52',
         'installation': [{'type': metadata_base.PrimitiveInstallationType.PIP,
                           'package_uri': 'git+https://github.com/Featuretools/ta1-primitives.git@{git_commit}#egg=featuretools_ta1-{version}'.format(
                               git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                               version=__version__
                            ),
                          }]
        })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed,
                         docker_containers=docker_containers)
        self._random_seed = random_seed
        self._include_unknown = hyperparams['include_unknown']
        self._top_n = hyperparams['top_n']
        self._remove_low_information = hyperparams['remove_low_information']

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
    def _update_metadata(
            cls,
            metadata: metadata_base.DataMetadata,
            features: list,
            for_value: DataFrame = None,
            source: typing.Any = None) -> metadata_base.DataMetadata:
        if source is None:
            source = cls
        resource_metadata = dict(metadata.query(()))
        new_metadata = metadata.clear(resource_metadata,
                                      for_value=for_value,
                                      source=source)
        new_metadata = cls._copy_elements_metadata(
            new_metadata, (), (), new_metadata, source=source)
        new_metadata = new_metadata.set_for_value(for_value)

        resource_metadata = dict(metadata.query((ALL_ELEMENTS,)))
        resource_metadata['dimension'] = {'length': len(features)}
        resource_metadata['ft_features'] = cloudpickle.dumps(features)
        new_metadata = new_metadata.update((ALL_ELEMENTS,),
                                            resource_metadata)

        for i, f in enumerate(features):
            resource_metadata = dict(metadata.query((ALL_ELEMENTS, i)))
            resource_metadata['semantic_types'] = [D3MMetadataTypes.to_d3m(
                f.variable_type)]
            resource_metadata['name'] = f.get_name()
            # TODO: structural type
            new_metadata = new_metadata.update((ALL_ELEMENTS, i),
                                               resource_metadata)
        return new_metadata

    def _get_fl_from_metadata(cls, metadata):
        feature_string = metadata.query((ALL_ELEMENTS,))['ft_features']
        return cloudpickle.loads(feature_string)

    @classmethod
    def can_accept(
            cls, *,
            method_name: str,
            arguments: typing.Dict[str,
                                   typing.Union[metadata_base.Metadata, type]]
            ) -> typing.Optional[metadata_base.DataMetadata]:
        output_metadata = super().can_accept(method_name=method_name,
                                             arguments=arguments)

        # If structural types didn't match, don't bother.
        if output_metadata is None:
            return None

        if 'inputs' not in arguments:
            return output_metadata

        inputs_metadata = typing.cast(
            metadata_base.DataMetadata, arguments['inputs'])
        fl = cls._get_fl_from_metadata(inputs_metadata)

        n_features = inputs_metadata.query(
            (ALL_ELEMENTS,))['dimension']['length']
        if len(fl) != n_features:
            return None

        return output_metadata

    def produce(self, *,
                inputs: Input,
                timeout: float=None,
                iterations: int=None) -> CallResult[Output]:
        fm = inputs
        fl = self._get_fl_from_metadata(fm.metadata)
        fm, fl = ft.encode_features(fm, fl,
                                    include_unknown=self._include_unknown,
                                    top_n=self._top_n)
        fm_with_metadata = DataFrame(fm)
        fm_with_metadata.metadata = self._update_metadata(
            metadata=inputs.metadata,
            features=fl,
            for_value=fm_with_metadata,
            source=self)
        return CallResult(fm_with_metadata)
