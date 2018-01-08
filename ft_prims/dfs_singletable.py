from typing import NamedTuple, Sequence, Any, List, Dict, Union
from collections import OrderedDict

import sklearn

from sklearn.ensemble.weight_boosting import AdaBoostClassifier

from d3m_metadata.container.numpy import ndarray
from d3m_metadata import hyperparams, params, metadata as metadata_module
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from primitive_interfaces.base import CallResult
from d3m_metadata import container
import featuretools as ft

# These are just regular Python variables
Inputs = container.dataset.Dataset
Outputs = container.pandas.DataFrame



# A named tuple for parameters. 
class Params(params.Params):
    pass


class Hyperparams(hyperparams.Hyperparams):
    max_depth = hyperparams.UniformInt(
                            lower=-1,
                            upper=5,
                            default=2,
                            description=''
                            )


class DFSSingle(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive wrapping featuretools on single table datasets
    """

    __author__ = "MIT/Feature Labs Team"
    metadata = metadata_module.PrimitiveMetadata(
        {'algorithm_types': ['DEEP_FEATURE_SYNTHESIS', ],
         'installation': [{'type': 'PIP', 'package_uri': 'git+https://gitlab.datadrivendiscovery.org/MIT-FeatureLabs/ta1-primitive.git@v0.1.0'}],
         'name': 'featuretools.dfs',
         'primitive_family': 'FEATURE_CONSTRUCTION',
         'python_path': 'd3m.primitives.ft_prims.dfs_singletable',
         'source': {'name': 'MIT/Feature Labs'},
         'version': '0.1.0',
         'id': '437da2ac-3c55-37a2-96e8-135e8e061182'})

    # It is important that all hyper-parameters (parameters which do not change during
    # a life-cycle of a primitive) are explicitly listed and typed in the constructor.
    # This allows one to do hyper-parameter tuning and explores the space of
    # hyper-parameters.
    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, str] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._clf = ft.dfs
        
        self._training_inputs = None
        self._training_outputs = None
        self._fitted = False

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._training_inputs = inputs
        self._training_outputs = outputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        # If already fitted with current training data, this call is a noop.
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None or self._training_outputs is None:
            raise ValueError("Missing training data.")

        self._clf.fit(self._training_inputs, self._training_outputs)
        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> Outputs:
        return CallResult(self._clf.predict(inputs))

    def get_params(self) -> Params:
        return Params(
                    estimators=self._clf.estimators_,
                    classes=self._clf.classes_,
                    n_classes=self._clf.n_classes_,
                    estimator_weights=self._clf.estimator_weights_,
                    estimator_errors=self._clf.estimator_errors_,
                    feature_importances=self._clf.feature_importances_)

    def set_params(self, *, params: Params) -> None:
        self._clf.estimators_ = params.estimators
        self._clf.classes_ = params.classes
        self._clf.n_classes_ = params.n_classes
        self._clf.estimator_weights_ = params.estimator_weights
        self._clf.estimator_errors_ = params.estimator_errors
        
