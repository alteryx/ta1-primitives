from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.primitives.feature_construction.deep_feature_synthesis import MultiTableFeaturization
from d3m.primitives.data_transformation import column_parser
import os


def generate_only():
    # Creating pipeline
    pipeline_description = Pipeline()
    pipeline_description.add_input(name='inputs')

    # Step 0: Parse columns
    step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.operator.dataset_map.DataFrameCommon'))
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_hyperparameter(name='primitive', argument_type=ArgumentType.VALUE, data=column_parser.Common)
    step_0.add_hyperparameter(name='resources', argument_type=ArgumentType.VALUE, data='all')
    step_0.add_hyperparameter(name='fit_primitive', argument_type=ArgumentType.VALUE, data='no')
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    # Step 1: MultiTableFeaturization
    step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization'))
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)

    # Step 2: learn model
    step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.regression.xgboost_gbtree.Common'))
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_output('produce')
    pipeline_description.add_step(step_2)

    # step 3: construct output
    step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
    step_3.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_3.add_output('produce')
    pipeline_description.add_step(step_3)

    # Final Output
    pipeline_description.add_output(name='output predictions', data_reference='steps.3.produce')

    # Generate .yml file for the pipeline
    import featuretools_ta1
    from pipeline_tests.utils import generate_pipeline

    dataset_name = 'kaggle_music_hackathon_MIN_METADATA'
    dataset_path = '/featuretools_ta1/datasets/seed_datasets_current'
    primitive_name = 'd3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization'
    version = featuretools_ta1.__version__
    test_name = os.path.splitext(os.path.basename(__file__))[0][5:]
    yml, pipeline_run_file = generate_pipeline(primitive_name=primitive_name,
                                               pipeline_description=pipeline_description,
                                               dataset_name=dataset_name,
                                               test_name=test_name)

    # fit-score command
    fs_cmd = 'python3 -m d3m runtime -d /featuretools_ta1/datasets/ fit-score -p {}'.format(yml)
    fs_cmd += ' -r {}/{}/{}_problem/problemDoc.json'.format(dataset_path, dataset_name, dataset_name)
    fs_cmd += ' -i {}/{}/TRAIN/dataset_TRAIN/datasetDoc.json'.format(dataset_path, dataset_name)
    fs_cmd += ' -t {}/{}/TEST/dataset_TEST/datasetDoc.json'.format(dataset_path, dataset_name)
    fs_cmd += ' -a {}/{}/SCORE/dataset_SCORE/datasetDoc.json'.format(dataset_path, dataset_name)
    fs_cmd += ' -O {}'.format(pipeline_run_file)

    # Run pipeline to save pipeline_run file
    os.system(fs_cmd)

    # Create and return command for running from pipeline_run file:
    pipeline_run_cmd = 'python3 -m d3m --pipelines-path /featuretools_ta1/MIT_FeatureLabs/{}/{}/pipelines/'.format(primitive_name, version)
    pipeline_run_cmd += ' runtime -d /featuretools_ta1/datasets/ fit-score'
    pipeline_run_cmd += ' -u {}'.format(pipeline_run_file)

    return pipeline_run_cmd


if __name__ == "__main__":
    pipeline_run_cmd = generate_only()
    # Run pipeline from pipeline run file
    # os.system(pipeline_run_cmd)