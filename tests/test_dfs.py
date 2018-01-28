from featuretools_ta1.dfs import DFS
from featuretools.primitives import make_trans_primitive, make_agg_primitive
import featuretools.variable_types as vtypes
import numpy as np
from d3m_metadata.container.dataset import D3MDatasetLoader
from d3m_metadata.problem import parse_problem_description
import json
import cloudpickle


# Initialize primitive on dataset

with open("config.json", 'r') as input_file:
    json_config = json.load(input_file)
train_uri = json_config['train_data']
test_uri = json_config['test_data']
problem_uri = json_config['problem']

train_ds = D3MDatasetLoader().load(dataset_uri=train_uri)
test_ds = D3MDatasetLoader().load(dataset_uri=test_uri)
problem = parse_problem_description(problem_uri)
target = problem['inputs'][0]['targets'][0]
metadata = DFS.metadata.query()['primitive_code']
hyperparams_class = metadata['class_type_arguments']['Hyperparams']
hp = hyperparams_class(hyperparams_class.defaults(),
                       normalize_categoricals_if_single_table=True)
hp_no_low_info = hyperparams_class(hyperparams_class.defaults(),
                                    remove_low_information=False)
hp_sample = hyperparams_class(hyperparams_class.defaults(),
                              sample_learning_data=10)

Log = make_trans_primitive(lambda array: np.log(array),
                           input_types=[vtypes.Numeric],
                           return_type=vtypes.Numeric,
                           name='log',
                           description='Log')

def absolute_max_min(array):
    return max(abs(array.max()), abs(array.min()))

AbsoluteMaxMin = make_agg_primitive(absolute_max_min,
                                    input_types=[vtypes.Numeric],
                                    return_type=vtypes.Numeric,
                                    name='absolute_max_min',
                                    description='AbsoluteMaxMin')
hp_custom = hyperparams_class(hyperparams_class.defaults(),
                              trans_primitives=[Log],
                              agg_primitives=[AbsoluteMaxMin])
dfs = DFS(hyperparams=hp)
dfs_2 = DFS(hyperparams=hp_no_low_info)
dfs_3 = DFS(hyperparams=hp_sample)
dfs_custom = DFS(hyperparams=hp_custom)



# Run primitive
dfs.set_training_data(inputs=[train_ds, target])
dfs_2.set_training_data(inputs=[train_ds, target])
dfs_3.set_training_data(inputs=[train_ds, target])
dfs_custom.set_training_data(inputs=[train_ds, target])
dfs.fit()
dfs_2.fit()
dfs_3.fit()
dfs_custom.fit()
train_feature_matrix = dfs.produce(inputs=[train_ds, target]).value
train_feature_matrix_low_info, _ = dfs_2.produce_encoded(inputs=[train_ds, target]).value
train_feature_matrix_sample = dfs_3.produce(inputs=[train_ds, target]).value
assert train_feature_matrix_sample.shape[0] == 10
train_feature_matrix_encoded, fl1 = dfs.produce_encoded(inputs=[train_ds, target]).value
assert train_feature_matrix_low_info.shape[1] != train_feature_matrix_encoded.shape[1]

fm_custom = dfs_custom.produce(inputs=[train_ds, target]).value
assert fm_custom.shape == (1073, 48)

# get_params
params = dfs.get_params()

# set_params
dfs = DFS(hyperparams=hp)

# Run primitive again
dfs.set_training_data(inputs=[train_ds, target])
dfs.set_params(params=params)
dfs.fit()
train_feature_matrix2 = dfs.produce(inputs=[train_ds, target]).value
train_feature_matrix_encoded2, fl2 = dfs.produce_encoded(inputs=[train_ds, target]).value
assert train_feature_matrix2.reset_index('time', drop=True).equals(train_feature_matrix.reset_index('time', drop=True))
assert train_feature_matrix_encoded2.reset_index('time', drop=True).equals(train_feature_matrix_encoded.reset_index('time', drop=True))


# Pickle primitive
pickled_dfs = cloudpickle.dumps(dfs)
# Unpickle primitive
unpickled_dfs = cloudpickle.loads(pickled_dfs)

assert unpickled_dfs.hyperparams == dfs.hyperparams
# Run primitive again
train_feature_matrix3 = unpickled_dfs.produce(inputs=[train_ds, target]).value
train_feature_matrix_encoded3, fl3 = unpickled_dfs.produce_encoded(inputs=[train_ds, target]).value
# time is current time, so will be different
assert train_feature_matrix3.reset_index('time', drop=True).equals(train_feature_matrix.reset_index('time', drop=True))
assert train_feature_matrix_encoded3.reset_index('time', drop=True).equals(train_feature_matrix_encoded.reset_index('time', drop=True))

# Run primitive again on new data
test_feature_matrix = unpickled_dfs.produce(inputs=[test_ds, target]).value
print("Test feature matrix")
print(test_feature_matrix.iloc[:5, :5])
