from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Dict
from d3m_metadata import metadata as metadata_module
from d3m.primitives.sklearn_wrap import SKRandomForestRegressor
from featuretools_ta1.rf_selector_base import (Params as BaseParams,
                                               SELECT_N_FEATURES as base_select_n_features,
                                               METADATA as BASE_METADATA,
                                               __author__ as base_author,
                                               RFFeatureSelectorBase,
                                               Inputs,
                                               Outputs)
from primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from primitive_interfaces.base import CallResult
import copy

metadata = SKRandomForestRegressor.metadata.query()['primitive_code']
SKRandomForestRegressorHP = metadata['class_type_arguments']['Hyperparams']
SKRandomForestRegressorParams = metadata['class_type_arguments']['Params']


class Params(BaseParams):
    selector_params: SKRandomForestRegressorParams
    selector_hyperparams: SKRandomForestRegressorHP


class Hyperparams(SKRandomForestRegressorHP):
    select_n_features = base_select_n_features


metadata = copy.deepcopy(BASE_METADATA)
metadata['description'] = "Feature selector using the Random Forest classifier's built in feature importances"
metadata['id'] = '1dcc7caa-fb9f-4e6f-b95d-359065dfc8d0'
metadata['python_path'] = 'd3m.primitives.featuretools_ta1.RFRegressorFeatureSelector'
metadata['name'] = 'Random Forest Regressor Feature Selector'


class RFRegressorFeatureSelector(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
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
        sk_rf_hp = SKRandomForestRegressorHP(SKRandomForestRegressorHP.defaults(),
                                              **sk_rf_hp)
        self._selector_object = SKRandomForestRegressor
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
