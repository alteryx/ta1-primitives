from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Union
from d3m.container.pandas import DataFrame
from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.metadata import hyperparams, params, base as metadata_module
from d3m import utils
from d3m.primitive_interfaces.base import CallResult
import os
import pandas as pd
import numpy as np
from . import __version__

# First element is D3MDataset, second element is dict of a target from problemDoc.json
Inputs = Union[DataFrame, d3m_ndarray]
Outputs = d3m_ndarray


class Params(params.Params):
    selector_object: object
    selector_fitted: bool
    training_inputs: Union[pd.DataFrame, np.ndarray]
    training_outputs: np.ndarray
    feature_importances: pd.DataFrame
    top_features: np.ndarray
    input_was_ndarray: bool


SELECT_N_FEATURES = hyperparams.UniformInt(
        default=50,
        lower=1,
        upper=500,
        description='Number of features to select'
    )

# For a list of options for each of these fields, see
# https://metadata.datadrivendiscovery.org/
METADATA = {'algorithm_types': ['RANDOM_FOREST', ],
     'primitive_family': 'FEATURE_SELECTION',
     "source": {
       "name": "MIT_FeatureLabs",
       "contact": "mailto://ben.schreck@featurelabs.com",
       "uris": ["https://doc.featuretools.com"],
       "license": "BSD-3-Clause"

     },
     "id": "replace-id",
     "name": "replace-name",
     "python_path": "d3m.primitives.featuretools_ta1.rf_selector_base",
     "keywords": ["feature selection"],
     "hyperparameters_to_tune": ["select_n_features", "n_estimators"],
     'version': __version__,
     'installation': [
        {
            "type": metadata_module.PrimitiveInstallationType.PIP,
            "package_uri": "git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@dist#egg=sklearn_wrap-0.1.1"
        },
        {'type': metadata_module.PrimitiveInstallationType.PIP,
                       'package_uri': 'git+https://github.com/Featuretools/ta1-primitives.git@{git_commit}#egg=featuretools_ta1-{version}'.format(
                           git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                           version=__version__
                        ),
        }]
}

__author__ = 'Feature Labs D3M team (Ben Schreck <ben.schreck@featurelabs.com>)'
class RFFeatureSelectorBase(object):
    """
    Feature selector using the Random Forest's built in feature importances
    """
    __author__ = 'Feature Labs D3M team (Ben Schreck <ben.schreck@featurelabs.com>)'

    # Output type for this needs to be specified (and should be None)
    def __init__(self, *,
                 hyperparams=None,
                 random_seed: int = 0):

        self._select_n_features = hyperparams['select_n_features']

        self._training_inputs = None
        self._training_outputs = None
        self._feature_importances = None
        self._top_features = None
        self._input_was_ndarray = False

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._input_was_ndarray = False
        if isinstance(inputs, np.ndarray):
            inputs = pd.DataFrame(inputs)
            self._input_was_ndarray = True

        self._training_inputs = inputs
        self._training_outputs = outputs
        self._selector.set_training_data(inputs=inputs, outputs=outputs)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._selector._fitted:
            return CallResult(None)
        self._selector.fit()

        self._feature_importances = pd.DataFrame({'Importance': self._selector._clf.feature_importances_,
                                                  'Feature': self._training_inputs.columns}).sort_values('Importance',
                                                                                                   ascending=False)
        self._top_features = self._feature_importances.iloc[:self._select_n_features]['Feature'].values

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if self._input_was_ndarray:
            inputs = pd.DataFrame(inputs)
        selected = inputs.loc[:, self._top_features]
        if self._input_was_ndarray:
            selected = selected.values
        return CallResult(selected)

    def get_params(self, Params):
        return Params(
            selector_params=self._selector.get_params(),
            selector_hyperparams=self._selector.hyperparams,
            selector_object=type(self._selector),
            selector_fitted=self._selector._fitted,
            training_inputs=self._training_inputs,
            training_outputs=self._training_outputs,
            feature_importances=self._feature_importances,
            top_features=self._top_features,
            input_was_ndarray=self._input_was_ndarray
        )

    def set_params(self, *, params) -> None:
        self._selector = params['selector_object'](hyperparams=params['selector_hyperparams'])
        self._selector.set_training_data(inputs=params['training_inputs'], outputs=params['training_outputs'])
        self._selector.set_params(params=params['selector_params'])
        if params['selector_fitted']:
            self._selector.fit()
        self._training_inputs = params['training_inputs']
        self._training_outputs = params['training_outputs']
        self._feature_importances = params['feature_importances']
        self._top_features = params['top_features']
        self._input_was_ndarray = params['input_was_ndarray']

    def __getstate__(self):
        return {'params': self.get_params(),
                'hyperparams': self.hyperparams,
                'random_seed': self.random_seed}

    def __setstate__(self, d, init_method):
        self.set_params(params=d['params'])
        init_method(hyperparams=d['hyperparams'],
                    random_seed=d['random_seed'],
                    docker_containers=None)
        self._select_n_features = self.hyperparams['select_n_features']
