from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Dict
from d3m import metadata as metadata_module
from d3m.metadata import hyperparams
from sklearn_wrap.SKRandomForestClassifier import SKRandomForestClassifier
from featuretools_ta1.rf_selector_base import (Params as BaseParams,
                                               SELECT_N_FEATURES as base_select_n_features,
                                               METADATA as BASE_METADATA,
                                               __author__ as base_author,
                                               RFFeatureSelectorBase,
                                               Inputs,
                                               Outputs)
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
import copy

metadata = SKRandomForestClassifier.metadata.query()['primitive_code']
SKRandomForestClassifierHP = metadata['class_type_arguments']['Hyperparams']
SKRandomForestClassifierParams = metadata['class_type_arguments']['Params']


class Params(BaseParams):
    selector_params: SKRandomForestClassifierParams
    selector_hyperparams: SKRandomForestClassifierHP


class Hyperparams(hyperparams.Hyperparams):
    select_n_features = copy.deepcopy(base_select_n_features)


for hp_name, hp in SKRandomForestClassifierHP.configuration.items():
    Hyperparams.configuration[hp_name] = hp


metadata = copy.deepcopy(BASE_METADATA)
metadata['description'] = "Feature selector using the Random Forest classifier's built in feature importances"
metadata['id'] = 'eb829ff3-bb50-4ec5-a21c-db32dc7d17e7'
metadata['python_path'] = 'd3m.primitives.featuretools_ta1.RFClassifierFeatureSelector'
metadata['name'] = 'Random Forest Classifier Feature Selector'


class RFClassifierFeatureSelector(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Feature selector using the Random Forest classifier's built in feature importances
    """
    __author__ = base_author
    metadata = metadata_module.PrimitiveMetadata(metadata)

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, str] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed,
                         docker_containers=docker_containers)

        hp_dict = dict(hyperparams)
        sk_rf_hp = {k: v for k, v in hp_dict.items()
                    if k != 'select_n_features'}
        sk_rf_hp = SKRandomForestClassifierHP(SKRandomForestClassifierHP.defaults(),
                                              **sk_rf_hp)
        self._selector_object = SKRandomForestClassifier
        self._selector = self._selector_object(hyperparams=sk_rf_hp, random_seed=random_seed)
        RFFeatureSelectorBase.__init__(self, hyperparams=hyperparams)

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        RFFeatureSelectorBase.set_training_data(self, inputs=inputs, outputs=outputs)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        return RFFeatureSelectorBase.fit(self, timeout=timeout, iterations=iterations)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        return RFFeatureSelectorBase.produce(self, inputs=inputs, timeout=timeout, iterations=iterations)

    def get_params(self) -> Params:
        return RFFeatureSelectorBase.get_params(self, Params)

    def set_params(self, *, params: Params) -> None:
        RFFeatureSelectorBase.set_params(self, params=params)

    def __getstate__(self):
        return RFFeatureSelectorBase.__getstate__(self)

    def __setstate__(self, d):
        RFFeatureSelectorBase.__setstate__(self, d, super().__init__)
        return
