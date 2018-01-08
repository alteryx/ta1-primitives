from typing import NamedTuple, Sequence, Any, List, Dict, Union
from collections import OrderedDict

import sklearn

from d3m_metadata.container.numpy import ndarray
from d3m_metadata import hyperparams, params, metadata as metadata_module
from primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from primitive_interfaces.base import CallResult
from d3m_metadata import container
import featuretools as ft
from .d3m_to_entityset import convert_d3m_dataset_to_entityset



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
        d3m_ds = inputs
        self.target = d3m_ds.problem.get_targets()[0]
        self.entityset, self.target_entity = convert_d3m_dataset_to_entityset(d3m_ds.dataset)

        self._training_inputs = inputs
        self._training_outputs = outputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        if self.entityset is None:
            raise ValueError("Must call .set_training_data() before calling .fit()")
        # If already fitted with current training data, this call is a noop.
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None:
            raise ValueError("Missing training data.")
        time_index = self.entityset[self.target_entity].time_index
        index = self.entityset[self.target_entity].index
        cutoff_time = None
        if time_index:
            cutoff_time = self.entityset[self.target_entity].df[[index, time_index]]
        self.features = ft.dfs(entityset=self.entityset,
                               target_entity=self.target_entity,
                               cutoff_time=cutoff_time,
                               features_only=True,
                               **self.dfs_kwargs)

        self._clf.fit(self._training_inputs, self._training_outputs)
        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> Outputs:

        if self.features is None:
            raise ValueError("Must call fit() before calling produce()")
        features = self.features
        d3m_ds = inputs
        entityset, target_entity = convert_d3m_dataset_to_entityset(d3m_ds.dataset)

        feature_matrix = ft.calculate_feature_matrix(features,
                                                     entityset=entityset,
                                                     **self.cfm_kwargs)
        for f in features:
            if issubclass(f.variable_type, vtypes.Discrete):
                feature_matrix[f.get_name()] = feature_matrix[f.get_name()].astype(object)
            elif issubclass(f.variable_type, vtypes.Numeric):
                feature_matrix[f.get_name()] = pd.to_numeric(feature_matrix[f.get_name()])
            elif issubclass(f.variable_type, vtypes.Datetime):
                feature_matrix[f.get_name()] = pd.to_datetime(feature_matrix[f.get_name()])

        return CallResult(feature_matrix)

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        pass 
        
