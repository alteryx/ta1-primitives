import featuretools_ta1


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

    with open(pipeline_description_meta, "w") as out:
        out.write(meta)


def generate_pipeline(primitive_name, pipeline_description, dataset_name):
    print("Generating pipeline for '{}'".format(dataset_name))

    pipeline_description_yml = "/featuretools_ta1/MIT_FeatureLabs/{primitive_name}/{version}/pipelines/{description}.yml".format(primitive_name=primitive_name, version=featuretools_ta1.__version__, description=pipeline_description.id)
    write_pipeline_yml(pipeline_description, pipeline_description_yml)
    write_pipeline_meta(primitive_name, pipeline_description, dataset_name)

    return
