from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.primitives.classification.random_forest import SKlearn as SKlearnRFC
from d3m.primitives.data_cleaning.imputer import SKlearn as SKlearnImputer
from d3m.primitives.data_transformation.construct_predictions import DataFrameCommon
from d3m.primitives.feature_construction.deep_feature_synthesis import Featuretools


def build_demo_pipeline():

    # Creating pipeline
    pipeline = Pipeline(context=Context.TESTING)
    pipeline.add_input(name='inputs')

    # Step 0: DFS
    step_0 = PrimitiveStep(primitive_description=Featuretools.metadata.query())
    step_0.add_argument(
        name='inputs',
        argument_type=ArgumentType.CONTAINER,
        data_reference='inputs.0'
    )
    step_0.add_output('produce')
    pipeline.add_step(step_0)

    # Step 1: SKlearnImputer
    step_1 = PrimitiveStep(primitive_description=SKlearnImputer.metadata.query())
    step_1.add_argument(
        name='inputs',
        argument_type=ArgumentType.CONTAINER,
        data_reference='steps.0.produce'
    )
    step_1.add_output('produce')
    pipeline.add_step(step_1)

    # Step 2: SKlearnRFC
    step_2 = PrimitiveStep(primitive_description=SKlearnRFC.metadata.query())
    step_2.add_hyperparameter(
        name='use_semantic_types',
        argument_type=ArgumentType.VALUE,
        data=True
    )
    step_2.add_hyperparameter(
        name='add_index_columns',
        argument_type=ArgumentType.VALUE,
        data=True
    )
    step_2.add_argument(
        name='inputs',
        argument_type=ArgumentType.CONTAINER,
        data_reference='steps.1.produce'
    )
    step_2.add_argument(
        name='outputs',
        argument_type=ArgumentType.CONTAINER,
        data_reference='steps.1.produce'
    )
    step_2.add_output('produce')
    pipeline.add_step(step_2)

    # Step 3: ConstructPredictions
    step_3 = PrimitiveStep(
        primitive_description=DataFrameCommon.metadata.query()
    )
    step_3.add_argument(
        name='inputs',
        argument_type=ArgumentType.CONTAINER,
        data_reference='steps.2.produce'
    )
    step_3.add_argument(
        name='reference',
        argument_type=ArgumentType.CONTAINER,
        data_reference='steps.2.produce'
    )
    step_3.add_output('produce')
    pipeline.add_step(step_3)

    # Final Output
    pipeline.add_output(name='output predictions', data_reference='steps.3.produce')

    return pipeline
