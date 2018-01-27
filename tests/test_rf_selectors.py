from featuretools_ta1.rf_clf_selector import RFClassifierFeatureSelector
from featuretools_ta1.rf_reg_selector import RFRegressorFeatureSelector
import cloudpickle
import numpy as np
import pandas as pd


def test_selector(SelectorClass):
    n_features = 200
    select_n_features = 100
    n_rows = 200
    feature_array = np.random.random(size=(n_rows, n_features))
    feature_df = pd.DataFrame(feature_array, columns=[str(i) for i in range(n_features)])
    target_array = np.random.randint(2, size=n_rows)

    metadata = SelectorClass.metadata.query()['primitive_code']
    SelectorHP = metadata['class_type_arguments']['Hyperparams']
    selector_hp = SelectorHP(SelectorHP.defaults(), select_n_features=select_n_features)
    selector = SelectorClass(hyperparams=selector_hp, random_seed=0)

    # test with ndarray
    selector.set_training_data(inputs=feature_array, outputs=target_array)
    selector.fit()
    selected = selector.produce(inputs=feature_array).value
    assert selected.shape == (n_rows, select_n_features)
    assert isinstance(selected, np.ndarray)

    # test with df
    selector.set_training_data(inputs=feature_df, outputs=target_array)
    selector.fit()
    selected = selector.produce(inputs=feature_df).value
    assert selected.shape == (n_rows, select_n_features)
    assert isinstance(selected, pd.DataFrame)
    assert all([t in selected for t in selector._top_features])


    # get_params
    params = selector.get_params()

    # set_params
    selector = SelectorClass(hyperparams=selector_hp, random_seed=0)
    selector.set_training_data(inputs=feature_df, outputs=target_array)
    selector.set_params(params=params)

    # Run primitive again
    selector.fit()
    selected2 = selector.produce(inputs=feature_df).value
    assert selected2.equals(selected)


    # Pickle primitive
    pickled_selected = cloudpickle.dumps(selector)
    # Unpickle primitive
    unpickled_selector = cloudpickle.loads(pickled_selected)

    assert unpickled_selector._select_n_features == select_n_features

    # Run primitive again
    selected3 = unpickled_selector.produce(inputs=feature_df).value
    assert selected3.equals(selected)

print("testing classifier selector")
test_selector(RFClassifierFeatureSelector)
print("testing regressor selector")
test_selector(RFRegressorFeatureSelector)
