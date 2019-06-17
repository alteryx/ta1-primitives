from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.primitives.feature_construction.deep_feature_synthesis import SingleTableFeaturization

# -> dataset_to_dataframe -> column_parser -> extract_columns_by_semantic_types(attributes) -> imputer -> random_forest
#                                             extract_columns_by_semantic_types(targets)    ->            ^

# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

# Step 0: dataset_to_dataframe
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# Step 1: column_parser
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.DataFrameCommon'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

# Step 2: DFS Single Table
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_output('produce')
pipeline_description.add_step(step_2)


# Step 3: imputer
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference="steps.2.produce")
step_3.add_hyperparameter(name='use_semantic_types', argument_type=ArgumentType.VALUE, data=True)
step_3.add_output('produce')
pipeline_description.add_step(step_3)

step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.one_hot_encoder.DistilOneHotEncoder'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference="steps.3.produce")
step_4.add_output('produce')
pipeline_description.add_step(step_4)


# Step 4: learn model
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.classification.random_forest.SKlearn'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_5.add_output('produce')
pipeline_description.add_step(step_5)

# step 5: construct output
step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.DataFrameCommon'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_6.add_output('produce')
pipeline_description.add_step(step_6)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')

# Output to YAML
# print(pipeline_description.to_yaml())
pipeline_description_yml = "/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization/0.6.0/pipelines/%s.yml" % pipeline_description.id
with open(pipeline_description_yml, "w") as out:
    out.write(pipeline_description.to_yaml())

meta = """{
    "problem": "LL0_acled_reduced_problem",
    "full_inputs": [
        "LL0_acled_reduced_dataset"
    ],
    "train_inputs": [
        "LL0_acled_reduced_dataset_TRAIN"
    ],
    "test_inputs": [
        "LL0_acled_reduced_dataset_TEST"
    ],
    "score_inputs": [
        "LL0_acled_reduced_dataset_SCORE"
    ]
}
"""
pipeline_description_meta = "/featuretools_ta1/MIT_FeatureLabs/d3m.primitives.feature_construction.deep_feature_synthesis.SingleTableFeaturization/0.6.0/pipelines/%s.meta" % pipeline_description.id
with open(pipeline_description_meta, "w") as out:
    out.write(meta)





from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, pipeline as pipeline_module, problem
from d3m.container.dataset import Dataset
from d3m.runtime import Runtime

# Loading problem description.
problem_doc = "/featuretools_ta1/datasets/seed_datasets_current/LL0_acled_reduced/TRAIN/problem_TRAIN/problemDoc.json"
problem_description = problem.parse_problem_description(problem_doc)

# Loading dataset.
data_doc = "/featuretools_ta1/datasets/seed_datasets_current/LL0_acled_reduced/TRAIN/dataset_TRAIN/datasetDoc.json"
path = 'file://{uri}'.format(uri=data_doc)
dataset = Dataset.load(dataset_uri=path)

# Loading pipeline description file.

with open(pipeline_description_yml, 'r') as file:
    pipeline_description = pipeline_module.Pipeline.from_yaml(string_or_file=file)


# Creating an instance on runtime with pipeline description and problem description.
runtime = Runtime(pipeline=pipeline_description, problem_description=problem_description, context=metadata_base.Context.TESTING)

# Fitting pipeline on input dataset.
fit_results = runtime.fit(inputs=[dataset])
fit_results.check_success()

# Producing results using the fitted pipeline.
data_doc = "/featuretools_ta1/datasets/seed_datasets_current/LL0_acled_reduced/TEST/dataset_TEST/datasetDoc.json"
path = 'file://{uri}'.format(uri=data_doc)
test_dataset = Dataset.load(dataset_uri=path)

produce_results = runtime.produce(inputs=[test_dataset])
produce_results.check_success()

print(produce_results.values)


# python3 -m d3m runtime fit-produce -p /featuretools_ta1/dfs-random-forest-classifier.yml -r -i /featuretools_ta1/tests-data/datasets/boston_dataset_1/datasetDoc.json -t /featuretools_ta1/tests-data/datasets/boston_dataset_1/datasetDoc.json -o results.csv -O pipeline_run.yml