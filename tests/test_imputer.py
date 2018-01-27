from featuretools_ta1.imputer import Imputer
import cloudpickle
import numpy as np
import pandas as pd


n_features = 200
n_rows = 200
empty_df = pd.DataFrame(index=range(n_rows), columns=[str(i) for i in range(n_features)])
empty_df.iloc[0, :] = 1
empty_array = empty_df.values

metadata = Imputer.metadata.query()['primitive_code']
ImputerHP = metadata['class_type_arguments']['Hyperparams']
imputer_hp = ImputerHP(ImputerHP.defaults(), strategy='most_frequent')
imputer = Imputer(hyperparams=imputer_hp, random_seed=0)

# test with ndarray
imputed = imputer.produce(inputs=empty_array).value
assert pd.DataFrame(imputed).dropna().shape[0] == imputed.shape[0]
assert isinstance(imputed, np.ndarray)

# test with df
imputed = imputer.produce(inputs=empty_df).value
assert imputed.dropna().shape[0] == imputed.shape[0]
assert isinstance(imputed, pd.DataFrame)


# get_params
params = imputer.get_params()

# set_params
imputer = Imputer(hyperparams=imputer_hp, random_seed=0)

# Run primitive again
imputed2 = imputer.produce(inputs=empty_df).value
assert imputed2.equals(imputed)


# Pickle primitive
pickled_imputed = cloudpickle.dumps(imputer)
# Unpickle primitive
unpickled_imputer = cloudpickle.loads(pickled_imputed)

assert unpickled_imputer._strategy == 'most_frequent'

# Run primitive again
imputed3 = unpickled_imputer.produce(inputs=empty_df).value
assert imputed3.equals(imputed)

