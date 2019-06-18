import featuretools_ta1

from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, pipeline as pipeline_module, problem
from d3m.container.dataset import Dataset
from d3m.runtime import Runtime

def write_pipeline_yml(pipeline_description, pipeline_description_yml):
    with open(pipeline_description_yml, "w") as out:
        out.write(pipeline_description.to_yaml())


def write_pipeline_meta(primitive_name, pipeline_description, dataset_name):
    meta = """{{
        "problem": "{dataset_name}_problem",
        "full_inputs": [
            "{dataset_name}_dataset"
        ],
        "train_inputs": [
            "{dataset_name}_dataset_TRAIN"
        ],
        "test_inputs": [
            "{dataset_name}_dataset_TEST"
        ],
        "score_inputs": [
            "{dataset_name}_dataset_SCORE"
        ]
    }}
    """.format(dataset_name=dataset_name)

    pipeline_description_meta = "/featuretools_ta1/MIT_FeatureLabs/{primitive_name}/{version}/pipelines/{description}.meta".format(primitive_name=primitive_name, version=featuretools_ta1.__version__, description=pipeline_description.id)
    print(pipeline_description_meta)
    
    with open(pipeline_description_meta, "w") as out:
        out.write(meta)


def load_pipeline_description(pipeline_description_yml):
    with open(pipeline_description_yml, 'r') as file:
        pipeline_description = pipeline_module.Pipeline.from_yaml(string_or_file=file)

    return pipeline_description


def load_problem(dataset_name):
    problem_doc = "/featuretools_ta1/datasets/seed_datasets_current/{dataset_name}/TRAIN/problem_TRAIN/problemDoc.json".format(dataset_name=dataset_name)
    return problem.parse_problem_description(problem_doc)


def load_dataset(dataset_name):
    data_doc = "/featuretools_ta1/datasets/seed_datasets_current/{dataset_name}/TRAIN/dataset_TRAIN/datasetDoc.json".format(dataset_name=dataset_name)
    path = 'file://{uri}'.format(uri=data_doc)
    return Dataset.load(dataset_uri=path)


def test_pipeline(primitive_name, pipeline_description, dataset_name):
    print("Testing with dataset '{}'".format(dataset_name))
    
    pipeline_description_yml = "/featuretools_ta1/MIT_FeatureLabs/{primitive_name}/{version}/pipelines/{description}.yml".format(primitive_name=primitive_name, version=featuretools_ta1.__version__, description=pipeline_description.id)
    print(pipeline_description_yml)
    write_pipeline_yml(pipeline_description, pipeline_description_yml)
    write_pipeline_meta(primitive_name, pipeline_description, dataset_name)
    
    # Loading problem description
    problem_description = load_problem(dataset_name)

    # Loading dataset.
    dataset = load_dataset(dataset_name)

    # Loading pipeline description file.
    pipeline_description = load_pipeline_description(pipeline_description_yml)
    
    # Creating an instance on runtime with pipeline description and problem description.
    runtime = Runtime(pipeline=pipeline_description, problem_description=problem_description, context=metadata_base.Context.TESTING)

    # Fitting pipeline on input dataset.
    fit_results = runtime.fit(inputs=[dataset])
    fit_results.check_success()

    # Producing results using the fitted pipeline.
    data_doc = "/featuretools_ta1/datasets/seed_datasets_current/{dataset_name}/TEST/dataset_TEST/datasetDoc.json".format(dataset_name=dataset_name)
    path = 'file://{uri}'.format(uri=data_doc)
    test_dataset = Dataset.load(dataset_uri=path)

    produce_results = runtime.produce(inputs=[test_dataset])
    produce_results.check_success()

    print(produce_results.values)
