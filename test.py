from d3m.primitives.featuretools_ta1 import DFS
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
                       normalize_categoricals_if_single_table=False)
dfs = DFS(hyperparams=hp)


# Run primitive
dfs.set_training_data(inputs=[train_ds, target])
dfs.fit()
train_feature_matrix = dfs.produce(inputs=[train_ds, target]).value


# Pickle primitive
pickled_dfs = cloudpickle.dumps(dfs)
# Unpickle primitive
unpickled_dfs = cloudpickle.loads(pickled_dfs)

# Run primitive again
test_feature_matrix = unpickled_dfs.produce(inputs=[test_ds, target]).value
print("Test feature matrix")
print(test_feature_matrix.iloc[:5, :5])
