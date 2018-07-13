#TODO: merge most of common code from here and reg into single file
from __future__ import absolute_import, division, print_function, unicode_literals
from typing import List, Dict, Optional
from numpy import ndarray
from scipy import sparse
from collections import OrderedDict
import sklearn
import numpy
import typing
import copy
import inspect

from sklearn.ensemble.forest import RandomForestClassifier
from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, base as metadata_base
from d3m.primitive_interfaces.base import CallResult, DockerContainer
import common_primitives.utils as common_utils
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from featuretools_ta1.rf_selector_base import (METADATA as BASE_METADATA,
                                               __author__ as base_author)
from sklearn_wrap.SKRFE import SKRFE, Inputs, Outputs, Params as SKRFEParams, Hyperparams as SKRFEHP
from sklearn_wrap.SKRandomForestClassifier import Hyperparams as SKRandomForestClassifierHP
from sklearn.feature_selection import RFE

from . import __version__


class Hyperparams(SKRFEHP, SKRandomForestClassifierHP):
    step = hyperparams.Union(
        OrderedDict({
           "int": hyperparams.Bounded[int](
                     default=1,

                     lower=1,
                     upper=None,
           ),
           "float": hyperparams.Bounded[float](
                     default=0.5,

                     lower=0,
                     upper=1,
           ),
           "none": hyperparams.Hyperparameter[None](
                     default=None,

           ),
        }),
        default='int',
        description='If greater than or equal to 1, then `step` corresponds to the (integer) number of features to remove at each iteration. If within (0.0, 1.0), then `step` corresponds to the percentage (rounded down) of features to remove at each iteration. ',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


class Params(SKRFEParams):
    model_estimators_: Optional[List[sklearn.tree.DecisionTreeClassifier]]
    model_feature_importances_: Optional[ndarray]
    model_n_features_: Optional[int]
    model_n_outputs_: Optional[int]
    model_oob_score_: Optional[float]
    model_oob_prediction_: Optional[ndarray]


# metadata = SKRFE.metadata
# new_metadata_info = copy.deepcopy(BASE_METADATA)
# new_metadata_info['description'] = "SK RFE with Random Forest Classifier"
# new_metadata_info['id'] = 'f4206ec7-11b1-42bc-9c80-909767a92ad8'
# new_metadata_info['python_path'] = 'd3m.primitives.featuretools_ta1.SKRFERandomForestClassifier'
# new_metadata_info['name'] = 'SK RFE Random Forest Classifier'
# new_metadata = metadata.update(new_metadata_info)


class SKRFERandomForestClassifier(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    __author__ = base_author
    # metadata = new_metadata

    metadata = metadata_module.PrimitiveMetadata({
        'algorithm_types': ['FEATURE_SCALING'],
        'name': 'SK RFE Random Forest Classifier'
        'id': 'f4206ec7-11b1-42bc-9c80-909767a92ad8',
        'python_path': 'd3m.primitives.featuretools_ta1.SKRFERandomForestClassifier',
        'description': 'SK RFE with Random Forest Classifier',
        "primitive_family": "DATA_PREPROCESSING",
        "source": {
            "name": "MIT_FeatureLabs",
            "contact": "mailto:max.kanter@featurelabs.com",
            "license": "BSD-3-Clause"
        },
        "version": __version__,
        'installation': [{
            'type': metadata_module.PrimitiveInstallationType.PIP,
            'package_uri': (
                'git+https://github.com/Featuretools/ta1-primitives.git'
                '@{git_commit}#egg=featuretools_ta1-{version}'
            ).format(
                git_commit=utils.current_git_commit(
                    os.path.dirname(__file__)
                ),
                version=__version__
            ),
        }]
    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None,
                 _verbose: int = 1) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        rf_kwargs = {}
        rf_sig = inspect.signature(RandomForestClassifier.__init__).parameters
        for k in SKRandomForestClassifierHP.configuration:
            if k in rf_sig:
                rf_kwargs[k] = hyperparams[k]
        estimator = RandomForestClassifier(verbose=_verbose, **rf_kwargs)
        self._clf = RFE(
            estimator=estimator,
            n_features_to_select=hyperparams['n_features_to_select'],
            step=hyperparams['step'],
            verbose=_verbose
        )
        self._training_inputs = None
        self._training_outputs = None
        self._target_names = None
        self._training_indices = None
        self._fitted = False

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs, self._training_indices = self._get_columns_to_fit(inputs, self.hyperparams)
        self._training_outputs, self._target_names = self._get_targets(outputs, self.hyperparams)
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None or self._training_outputs is None:
            raise ValueError("Missing training data.")
        sk_training_output = d3m_ndarray(self._training_outputs)

        shape = sk_training_output.shape
        if len(shape) == 2 and shape[1] == 1:
            sk_training_output = numpy.ravel(sk_training_output)

        self._clf.fit(self._training_inputs, sk_training_output)
        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        sk_inputs = inputs
        original_semantic_types = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, i))['semantic_types']
                                   for i in range(len(inputs.columns))]
        original_semantic_types_training = [s for i, s in enumerate(original_semantic_types)
                                            if i in self._training_indices]
        original_other_semantic_types = [s for i, s in enumerate(original_semantic_types)
                                            if i not in self._training_indices]

        if self.hyperparams['use_semantic_types']:
            sk_inputs = inputs.iloc[:, self._training_indices]
        sk_output = self._clf.transform(sk_inputs)
        if sparse.issparse(sk_output):
            sk_output = sk_output.toarray()
        output = d3m_dataframe(sk_output, generate_metadata=False, source=self)

        other_input_columns = inputs.columns[[i for i in range(len(inputs.columns))
                                              if i not in self._training_indices]]
        for c in other_input_columns:
            output[c] = inputs[c].values
        output.metadata = inputs.metadata.clear(source=self, for_value=output, generate_metadata=True)
        semantic_types_to_update = [original_semantic_types_training[i] for i in range(len(self._training_indices))
                                    if self._clf.support_[i]]
        semantic_types_to_update += original_other_semantic_types

        output.metadata = self._add_semantic_types(metadata=output.metadata, semantic_types=semantic_types_to_update, source=self)
        # TODO combine columns based on 3 control hyperparameters.
        return CallResult(output)

    def get_params(self) -> Params:
        if not self._fitted:
            raise ValueError("Fit not performed.")
        return Params(
            model_estimators_=getattr(self._clf.estimator_, 'estimators_', None),
            model_feature_importances_=getattr(self._clf.estimator_, 'feature_importances_', None),
            model_n_features_=getattr(self._clf.estimator_, 'n_features_', None),
            model_n_outputs_=getattr(self._clf.estimator_, 'n_outputs_', None),
            model_oob_score_=getattr(self._clf.estimator_, 'oob_score_', None),
            model_oob_prediction_=getattr(self._clf.estimator_, 'oob_prediction_', None),
            n_features_=getattr(self._clf, 'n_features_', None),
            support_=getattr(self._clf, 'support_', None),
            ranking_=getattr(self._clf, 'ranking_', None),
            #training_inputs=self._training_inputs,
            #training_outputs=self._training_outputs,
            target_names_=self._target_names,
            training_indices_=self._training_indices,
            estimator_=None
        )

    def set_params(self, *, params: Params) -> None:
        self._clf.estimator.estimators_ = params['model_estimators_']
        self._clf.estimator.n_features_ = params['model_n_features_']
        self._clf.estimator.n_outputs_ = params['model_n_outputs_']
        self._clf.estimator.oob_score_ = params['model_oob_score_']
        self._clf.estimator.oob_prediction_ = params['model_oob_prediction_']
        self._clf.n_features_ = params['n_features_']
        self._clf.support_ = params['support_']
        self._clf.ranking_ = params['ranking_']
        #self._training_inputs=params['training_inputs']
        #self._training_outputs=params['training_outputs']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._fitted = True

    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: Hyperparams):
        if not hyperparams['use_semantic_types']:
            return inputs, [len(inputs.columns)]

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = common_utils.get_columns_to_use(inputs_metadata,
                                                                             use_columns=hyperparams['use_columns'],
                                                                             exclude_columns=hyperparams['exclude_columns'],
                                                                             can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce

    @classmethod
    def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int, hyperparams: Hyperparams) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        accepted_structural_types = [int, float, numpy.int64, numpy.float64]
        if column_metadata['structural_type'] not in accepted_structural_types:
            return False

        semantic_types = column_metadata.get('semantic_types', [])
        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        if "https://metadata.datadrivendiscovery.org/types/Attribute" in semantic_types:
            return True

        return False

    @classmethod
    def _get_targets(cls, data: d3m_dataframe, hyperparams: Hyperparams):
        if not hyperparams['use_semantic_types']:
            return data, []
        target_names = []
        target_column_indices = []
        metadata = data.metadata
        target_column_indices.extend(metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        target_column_indices.extend(metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/RedactedTarget'))
        target_column_indices.extend(
            metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget'))
        target_column_indices = list(set(target_column_indices))
        for column_index in target_column_indices:
            if column_index is metadata_base.ALL_ELEMENTS:
                continue
            column_index = typing.cast(metadata_base.SimpleSelectorSegment, column_index)
            column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            target_names.append(column_metadata.get('name', str(column_index)))

        targets = data.iloc[:, target_column_indices]
        return targets, target_names

    @classmethod
    def _add_semantic_types(cls, metadata: metadata_base.DataMetadata,
                            semantic_types,
                            source: typing.Any) -> metadata_base.DataMetadata:
        for column_index, stypes in enumerate(semantic_types):
            for s in stypes:
                metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                      s,
                                                      source=source)
        return metadata

SKRFE.__doc__ = RFE.__doc__
