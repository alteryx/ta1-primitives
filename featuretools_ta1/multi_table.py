from d3m.metadata import base as metadata_base, hyperparams, params
from d3m import container, exceptions
from d3m.base import utils as d3m_utils


from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from typing import Dict, Optional, Sequence, Any
from featuretools_ta1 import config as CONFIG
from d3m.primitive_interfaces.base import CallResult, DockerContainer, MultiCallResult
from d3m.exceptions import PrimitiveNotFittedError
from featuretools_ta1.utils import drop_percent_null, select_one_of_correlated
import featuretools_ta1
import featuretools as ft
import numpy as np
import typing
import pandas as pd
from featuretools_ta1.utils import add_metadata, find_primary_key, find_target_column, get_featuretools_variable_types
import featuretools_ta1.semantic_types as st

Inputs = container.Dataset
Outputs = container.DataFrame
TARGET_ENTITY = "table"


class Params(params.Params):
    # A named tuple for parameters.
    features: Optional[Sequence[Any]]
    _target_resource_id: Optional[str]

# todo target entity


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
    max_features = hyperparams.Hyperparameter[int](
        default=100,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Cap the number of generated features to this number. If -1, no limit."
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
            'hyperparameters_to_tune': ['max_percent_null', 'max_correlation', 'max_features'],
        },
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        self._fitted = False

        # chunk size for feature calculation
        self.chunk_size = .5

        # todo handle use_columns, exclude_columns
        # todo handle return result

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._target_resource_id, _ = d3m_utils.get_tabular_resource(inputs, self.hyperparams["target_resource"])
        # d3m.base.utils.
        self._inputs = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        es = self._make_entityset(self._inputs)

        ignore_variables = {}

        # if there is a target column on the target entity, ignore it
        target_column = find_target_column(self._inputs[self._target_resource_id], return_index=False)
        if target_column:
            ignore_variables = {self._target_resource_id: target_column}
        
        trans_primitives = ["is_weekend", "day", "month", "year", "week", "weekday", "num_words", "num_characters",
                            "add_numeric", "subtract_numeric", "multiply_numeric", "divide_numeric"]

        agg_primitives = ["mean", "sum", "count", "mode", "num_unique"]

        # generate all the features
        fm, features = ft.dfs(
            target_entity=self._target_resource_id,
            entityset=es,
            agg_primitives=agg_primitives,
            trans_primitives=trans_primitives,
            max_depth=self.hyperparams["max_depth"],
            chunk_size=self.chunk_size,
            ignore_variables=ignore_variables,
            max_features=self.hyperparams["max_features"],
        )

        # treat inf as null. repeat in produce step
        fm = fm.replace([np.inf, -np.inf], np.nan)

        # filter based on nulls and correlation
        fm, features = drop_percent_null(fm, features, max_percent_null=self.hyperparams['max_percent_null'])
        fm, features = select_one_of_correlated(fm, features, threshold=self.hyperparams['max_correlation'])

        self.features = features

        fm = add_metadata(fm, self.features)

        self._fitted = True

        return CallResult(fm)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        es = self._make_entityset(inputs.copy())

        fm = ft.calculate_feature_matrix(
            entityset=es,
            features=self.features,
            chunk_size=self.chunk_size
        )

        # make sure the feature matrix is ordered the same as the input
        fm = fm.reindex(es[self._target_resource_id].df.index)
        fm = fm.reset_index(drop=True)  # d3m wants index to increment by 1

        # treat inf as null like fit step
        fm = fm.replace([np.inf, -np.inf], np.nan)

        # todo add this metadata handle
        fm = add_metadata(fm, self.features)
        fm = self._add_labels(fm, inputs)

        return CallResult(fm)

    def _add_labels(self, fm, inputs):
        pk_index = find_primary_key(inputs[self._target_resource_id], return_index=True)

        # if a pk is found
        if pk_index is not None:
            pk_col = inputs[self._target_resource_id].select_columns([pk_index])
            fm = fm.append_columns(pk_col)

        target_index = find_target_column(inputs[self._target_resource_id], return_index=True)

        # if a target is found,
        if target_index is not None:
            labels = inputs[self._target_resource_id].select_columns(target_index)
            fm = fm.append_columns(labels)

        return fm

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                features=None,
                _target_resource_id=None)

        return Params(
            features=self.features,
            _target_resource_id=self._target_resource_id)

    def set_params(self, *, params: Params) -> None:
        self.features = params["features"]
        self._target_resource_id = params["_target_resource_id"]

        # infer if it is fitted
        if self.features is not None:
            self._fitted = True
        if self._target_resource_id is not None:
            self._fitted = True

    def _make_entityset(self, inputs):
        es = ft.EntitySet()
        resources = inputs.items()

        # relations is a dictionary mapping resource to
        # (other resource, direction (true if other resource is parent, false if child), key resource index, other resource index)
        relations = inputs.get_relations_graph()

        # Create a list to store relationships to add to entity set
        relationships_to_add = []

        for resource_id, resource_df in resources:
            # make sure resources is a dataframe
            if not isinstance(resource_df, container.DataFrame):
                continue

            primary_key = find_primary_key(resource_df)

            if primary_key is None:
                # if there is no primary key, skip the dataset
                continue
                # raise RuntimeError("Cannot find primary key in resource %s" % (str(resource_id)))

            cols_to_use = resource_df.metadata.list_columns_with_semantic_types([st.PRIMARY_KEY, st.ATTRIBUTE])

            resource_df = resource_df.select_columns(cols_to_use)

            variable_types = get_featuretools_variable_types(resource_df)

            # Get the columns used in relationships and store child to parent relationships
            relationships = [r for r in relations[resource_id]]
            relationship_cols = []
            for rel in relationships:
                parent_entity_id = rel[0]
                parent_variable_id = inputs.metadata.query([parent_entity_id, "ALL_ELEMENTS", rel[3]])["name"]
                child_entity_id = resource_id
                child_variable_id = inputs.metadata.query([child_entity_id, "ALL_ELEMENTS", rel[2]])["name"]
                relationship_cols = relationship_cols + [parent_variable_id, child_variable_id]
                # if this is child to parent, add data to create relationship later
                if rel[1]:
                    relationships_to_add.append({
                        'parent_entity': parent_entity_id,
                        'parent_var': parent_variable_id,
                        'child_entity': child_entity_id,
                        'child_var': child_variable_id,
                    })

            # cast objects to categories to reduce memory footprint
            for col in resource_df.select_dtypes(include='object'):
                # if the column is used in a relationship, don't cast
                if col in relationship_cols:
                    continue
                resource_df[col] = resource_df[col].astype("category")
                # todo file issue on FT github for this work around
                # basically some primitives try to do fillna("") on the category and this breaks
                if "" not in resource_df[col].cat.categories:
                    resource_df[col] = resource_df[col].cat.add_categories("")

            es.entity_from_dataframe(
                entity_id=resource_id,
                index=primary_key,
                dataframe=pd.DataFrame(resource_df),
                variable_types=variable_types
            )

        for rel in relationships_to_add:
            # make sure all columns used in relationships are cast properly - catches error in dataset types
            try:
                es[rel['parent_entity']].df[rel['parent_var']] = es[rel['parent_entity']].df[rel['parent_var']].astype("int")
                es[rel['child_entity']].df[rel['child_var']] = es[rel['child_entity']].df[rel['child_var']].astype("int")
            except:
                es[rel['parent_entity']].df[rel['parent_var']] = es[rel['parent_entity']].df[rel['parent_var']].astype("object")
                es[rel['child_entity']].df[rel['child_var']] = es[rel['child_entity']].df[rel['child_var']].astype("object")

            es.add_relationship(
                ft.Relationship(
                    es[rel['parent_entity']][rel['parent_var']],
                    es[rel['child_entity']][rel['child_var']],
                )
            )

        return es

    def fit_multi_produce(self, *, produce_methods: Sequence[str], inputs: Inputs, timeout: float = None, iterations: int = None) -> MultiCallResult:
        self.set_training_data(inputs=inputs)  # type: ignore

        method_name = produce_methods[0]
        if method_name != 'produce':
            raise exceptions.InvalidArgumentValueError("Invalid produce method name '{method_name}'.".format(method_name=method_name))

        fit_results = self.fit(timeout=timeout, iterations=iterations)
        fm = fit_results.value
        fm = fm.reset_index(drop=True)

        fm = self._add_labels(fm, inputs)

        result = CallResult(fm)

        return MultiCallResult(
            values={method_name: result.value},
        )
