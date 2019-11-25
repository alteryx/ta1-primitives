import featuretools_ta1


def write_pipeline_yml(pipeline_description, pipeline_description_yml):
    with open(pipeline_description_yml, "w") as out:
        out.write(pipeline_description.to_yaml())


def generate_pipeline(primitive_name, pipeline_description, dataset_name, test_name):
    print("Generating pipeline for '{}'".format(dataset_name))

    pipeline_description_yml = "/featuretools_ta1/MIT_FeatureLabs/{primitive_name}/{version}/pipelines/{description}.yml".format(primitive_name=primitive_name, version=featuretools_ta1.__version__, description=pipeline_description.id)
    write_pipeline_yml(pipeline_description, pipeline_description_yml)

    pipeline_run_file = "/featuretools_ta1/MIT_FeatureLabs/{primitive_name}/{version}/pipeline_runs/{test_name}_pipeline_run.yml".format(primitive_name=primitive_name, version=featuretools_ta1.__version__, test_name=test_name)

    return pipeline_description_yml, pipeline_run_file
