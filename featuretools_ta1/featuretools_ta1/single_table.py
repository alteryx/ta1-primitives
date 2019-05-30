from d3m.metadata import base as metadata_base, hyperparams
from d3m import container, exceptions, utils as d3m_utils
from d3m.primitive_interfaces.featurization import FeaturizationLearnerPrimitiveBase



Inputs = container.DataFrame
Outputs = container.DataFrame

class Hyperparams(hyperparams.Hyperparams):
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


class SingleTableDFS(transformer.FeaturizationLearnerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """TODO: Write documentation"""
    __author__ = 'Mingjie Sun <sunmj15@gmail.com>'
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'f31f8c1f-d1c5-43e5-a4b2-2ae4a761ef2e',
            'version': '1.0.0',
            'name': "Single Table Deep Feature Synthesis",
            'python_path': 'd3m.primitives.data_transformation.denormalize.Common',
            'source': {
                'name': config.author,
                'contact': config.contact,
                'uris': ['https://docs.featuretools.com'],
                'license': 'BSD-3-Clause'
            },
            'installation': [{
               'type': metadata_base.PrimitiveInstallationType.PIP,
               'package_uri': 'git+https://github.com/Featuretools/ta1-primitives.git@{git_commit}#egg=featuretools_ta1'.format(
                   git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
               ),
            }],
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
            'hyperparameters_to_tune': ['TODO'],
        },
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def set_training_data(self, *, inputs: Inputs) -> None:
        import pdb; pdb.set_trace()
        pass

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        pass

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        pass

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        pass
"""

do we need the docker containers input to init?

# def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
"""
