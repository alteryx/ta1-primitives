from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.primitives.feature_construction.deep_feature_synthesis import SingleTableFeaturization
from d3m.primitives.feature_construction.deep_feature_synthesis import MultiTableFeaturization
from d3m.primitives.data_transformation import column_parser
import os


def generate_only():
    # -> dataset_to_dataframe -> column_parser -> extract_columns_by_semantic_types(attributes) -> imputer -> random_forest
    #                                             extract_columns_by_semantic_types(targets)    ->            ^

    # Creating pipeline
    pipeline_description = Pipeline()
    pipeline_description.add_input(name='inputs')

    # Step 0: Parse columns
    step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.operator.dataset_map.DataFrameCommon'))
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_hyperparameter(name='primitive', argument_type=ArgumentType.VALUE,
                            data=column_parser.DataFrameCommon)
    step_0.add_hyperparameter(name='resources', argument_type=ArgumentType.VALUE,
                            data='all')
    step_0.add_hyperparameter(name='fit_primitive', argument_type=ArgumentType.VALUE,
                            data='no')
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    # Step 1: MultiTableFeaturization
    step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization'))
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


    # Step 4: learn model
    step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.classification.xgboost_gbtree.DataFrameCommon'))
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
    step_4.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_4.add_output('produce')
    pipeline_description.add_step(step_4)

    # step 5: construct output
    step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.DataFrameCommon'))
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
    step_5.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_5.add_output('produce')
    pipeline_description.add_step(step_5)

    # Final Output
    pipeline_description.add_output(name='output predictions', data_reference='steps.5.produce')

    # Generate .yml and .meta files for the pipeline
    from pipeline_tests.utils import generate_pipeline
    yml, meta = generate_pipeline(primitive_name='d3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization',
                                  pipeline_description = pipeline_description,
                                  dataset_name='LL1_retail_sales_multi')

    return yml, meta

if __name__ == "__main__":
    yml, meta = generate_only()
    print("Running test...")
    cmd = 'python3 -m d3m runtime -d /featuretools_ta1/datasets/ evaluate -m {} -p {} -d /pipeline_tests/kfold_pipeline.yml'.format(meta, yml)
    os.system(cmd)
