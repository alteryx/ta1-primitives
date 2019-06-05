from d3m.metadata import base as metadata_base, hyperparams, params
from d3m import container, exceptions
from d3m.base import utils as d3m_utils


from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from typing import Dict, Optional, Union
from featuretools_ta1 import config as CONFIG
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.exceptions import PrimitiveNotFittedError
from featuretools_ta1.utils import drop_percent_null, select_one_of_correlated
import featuretools_ta1
import featuretools as ft
import numpy as np
import typing

Inputs = container.Dataset
Outputs = container.DataFrame
TARGET_ENTITY = "table"

class Params(params.Params):
    # A named tuple for parameters.
    features: Optional[bytes]

# todo target entity

PRIMARY_KEY = "https://metadata.datadrivendiscovery.org/types/PrimaryKey"
TEXT = "http://schema.org/Text"
NUMBER = "http://schema.org/Number"
INTEGER = "http://schema.org/Integer"
FLOAT = "http://schema.org/Float"
DATETIME = "http://schema.org/DateTime"
BOOLEAN = "http://schema.org/Boolean"
CATEGORICAL = "https://metadata.datadrivendiscovery.org/types/CategoricalData"

class Hyperparams(hyperparams.Hyperparams):
    target_resource = hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="The resource to create features for. If \"None\" then it starts from the dataset entry point.",
    )
    max_depth = hyperparams.Hyperparameter[int](
        default=2,
        description='The maximum number of featuretools primitives to stack when creating features',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_percent_null = hyperparams.Bounded[float](
        default=.5,
        lower=0,
        upper=1,
        description='The maximum percentage of null values allowed in returned features. A lower value means features may have more null nulls.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_correlation = hyperparams.Bounded[float](
        default=.9,
        lower=0,
        upper=1,
        description='The maximum allowed correlation between any two features returned. A lower value means features will be more uncorrelated',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='new',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.",
    )


class MultiTableFeaturization(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """This primitive creates new interaction features for an input dataframe.

    After creating features it reduces the set of possible features using an unsupervised approach"""
    __author__ = 'Max Kanter <max.kanter@featurelabs.com>'
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'e659ef3a-f17c-4bbf-9e5a-13de79a4e55b',
            'version': featuretools_ta1.__version__,
            'name': "Multi Table Deep Feature Synthesis",
            'python_path': 'd3m.primitives.feature_construction.deep_feature_synthesis.MultiTableFeaturization',
            'source': {
                'name': CONFIG.AUTHOR,
                'contact': CONFIG.CONTACT,
                'uris': ['https://docs.featuretools.com'],
                'license': 'BSD-3-Clause'
            },
            'installation': CONFIG.INSTALLATION,
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DEEP_FEATURE_SYNTHESIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_CONSTRUCTION,
            'keywords': [
                'featurization',
                'feature engineering',
                'feature extraction',
                'feature construction'
            ],
            'hyperparameters_to_tune': ['max_percent_null', 'max_correlation'],
        },
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        self._fitted = False

        # todo handle use_columns, exclude_columns
        # todo handle return result

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._target_resource_id, _ = d3m_utils.get_tabular_resource(inputs, self.hyperparams["target_resource"])
        # d3m.base.utils.
        self._inputs = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:


        # todo ignore target columns
        es = self._make_entityset(self._inputs)

        import pandas as pd
        class NumCharacters(ft.primitives.base.transform_primitive_base.TransformPrimitive):
            """Calculates the number of characters in a string.

            Examples:
                >>> num_characters = NumCharacters()
                >>> num_characters(['This is a string',
                ...                 'second item',
                ...                 'final1']).tolist()
                [16, 11, 6]
            """
            name = 'num_characters'
            input_types = [ft.variable_types.Text]
            return_type = ft.variable_types.Numeric
            default_value = 1

            def get_function(self):
                def test(array):
                    import pdb; pdb.set_trace()
                    return pd.Series(array).fillna('').str.len()
                return test


        # generate all the features
        import pdb; pdb.set_trace()
        fm, features = ft.dfs(
            target_entity="1",
            entityset=es,
            agg_primitives=["mean", "sum", "count", "mode", "num_unique"],
            trans_primitives=["day", "week", "month", "year", "num_words", NumCharacters],
            max_depth=self.hyperparams["max_depth"],
        )

        import pdb; pdb.set_trace()
        # treat inf as null. repeat in produce step
        fm = fm.replace([np.inf, -np.inf], np.nan)

        # filter based on nulls and correlation
        fm, features = drop_percent_null(fm, features, max_percent_null=self.hyperparams['max_percent_null'])
        fm, features = select_one_of_correlated(fm, features, threshold=self.hyperparams['max_correlation'])

        self.features = features

        self._fitted = True


        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        es = self._make_entityset(inputs.copy())

        fm = ft.calculate_feature_matrix(
            entityset=es,
            features=self.features
        )

        # treat inf as null like fit step
        fm = fm.replace([np.inf, -np.inf], np.nan)

        outputs = container.DataFrame(fm, generate_metadata=True)

        return CallResult(fm)

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(features=None)

        return Params(features=None)

    def set_params(self, *, params: Params) -> None:
        self.features = params["features"]

        # infer if it is fitted
        if self.features:
            self._fitted = True


    def _make_entityset(self, input):
        es = ft.EntitySet()
        resources = self._inputs.items()
        for resource_id, resource_df in resources:
            # make sure resources is a dataframe
            if not isinstance(resource_df, container.DataFrame):
                continue


            num_columns = self._inputs.metadata.query([resource_id, "ALL_ELEMENTS"])["dimension"]["length"]

            variable_types = {}

            # find primary key and other variable types
            primary_key = None
            for i in range(num_columns):
                metadata = self._inputs.metadata.query([resource_id, "ALL_ELEMENTS", i])
                semantic_types = metadata["semantic_types"]
                col_name = metadata["name"]
                if PRIMARY_KEY in semantic_types:
                    primary_key = col_name
                elif TEXT in semantic_types:
                    variable_types[col_name] = ft.variable_types.Text
                elif NUMBER in semantic_types or FLOAT in semantic_types or INTEGER in semantic_types:
                    variable_types[col_name] = ft.variable_types.Numeric
                elif DATETIME in semantic_types:
                    variable_types[col_name] = ft.variable_types.Datetime
                elif CATEGORICAL in semantic_types:
                    variable_types[col_name] = ft.variable_types.Categorical

                # todo: this should probably be removed because type conversion should happen outside primitive
                resource_df[col_name] = resource_df[col_name].astype(metadata['structural_type'])

            # import pdb; pdb.set_trace()

            es.entity_from_dataframe(
                entity_id=resource_id,
                index=primary_key,
                dataframe=resource_df.head(5),
                variable_types=variable_types
            )

            # work around for featuretools converting dtypes
            # leading to error later when trying to add relationships
            es[resource_id].df = resource_df



        # relations is a dictionary mapping resource to
        # (other resource, direction (true if other resource is parent, false if child), key resource index, other resource index)
        relations = self._inputs.get_relations_graph()
        for entity in es.entities:
            # only want relationships in child to parent direction
            relationships = [r for r in relations[entity.id] if r[1]]

            for rel in relationships:
                parent_entity_id = rel[0]
                parent_variable_id = self._inputs.metadata.query([parent_entity_id, "ALL_ELEMENTS", rel[3]])["name"]
                child_entity_id = entity.id
                child_variable_id = self._inputs.metadata.query([child_entity_id, "ALL_ELEMENTS", rel[2]])["name"]
                es.add_relationship(
                    ft.Relationship(
                        es[parent_entity_id][parent_variable_id],
                        es[child_entity_id][child_variable_id]
                    )
                )

        return es


"""

do we need the docker containers input to init?

# def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
fix path

TODOS
* handle non numeric
"""
