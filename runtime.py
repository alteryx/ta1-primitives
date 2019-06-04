import argparse
import inspect
import json
import logging
import os
import os.path
import pickle
import sys
import tempfile
import traceback
import typing
import uuid
from urllib import parse as url_parse

import frozendict  # type: ignore
import pandas  # type: ignore

from d3m import container, deprecate, exceptions, types, utils
from d3m.container import dataset as dataset_module
from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, pipeline as pipeline_module, pipeline_run as pipeline_run_module, problem
from d3m.primitive_interfaces import base

logger = logging.getLogger(__name__)


class Result:
    """
    Results from running a pipeline.

    Parameters
    ----------
    pipeline_run : PipelineRun
        A pipeline run description.
    values : Dict[str, Any]
        A map between data references and their values computed during pipeline run.
    error : Optional[Exception]
        If during a run an exception occurred, then it is available here.
    """

    def __init__(self, pipeline_run: pipeline_run_module.PipelineRun, values: typing.Dict[str, typing.Any], error: Exception = None) -> None:
        self.pipeline_run = pipeline_run
        self.values = values
        self.error = error

    def has_error(self) -> bool:
        """
        Returns ``True`` if pipeline has not successfully finished.
        """

        return self.error is not None

    def check_success(self) -> None:
        """
        Throws an exception if pipeline has not successfully finished.
        """

        if self.has_error():
            raise self.error


class MultiResult(typing.List[Result]):
    """
    Results of running a pipeline multiple times.
    """

    @property
    def pipeline_runs(self) -> typing.Sequence[pipeline_run_module.PipelineRun]:
        return [result.pipeline_run for result in self]

    def has_error(self) -> bool:
        """
        Returns ``True`` if any of pipelines has not successfully finished.
        """

        return any(result.has_error() for result in self)

    def check_success(self) -> None:
        """
        Throws an exception if pipeline has not successfully finished in any of the runs.
        """

        for result in self:
            result.check_success()


# TODO: Add debug logging to the runtime.
class Runtime:
    """
    Reference runtime to fit and produce a pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        A pipeline to run.
    hyperparams : Sequence
        Values for free hyper-parameters of the pipeline. It should be a list, where each element corresponds
        to free hyper-parameters of the corresponding pipeline step. Not all free hyper-parameters have to be
        specified. Default values are used for those which are not. Optional.
    problem_description : Problem
        A parsed problem description in standard problem description schema.
    context : Context
        In which context to run pipelines, default is ``TESTING``.
    random_seed : int
        A random seed to use for every run. This control all randomness during the run.
    volumes_dir : str
        Path to a directory with static files required by primitives.
        In the standard directory structure (as obtained running ``python3 -m d3m.index download``).
    scratch_dir : str
        Path to a directory to store any temporary files needed during execution.
    is_standard_pipeline : bool
        Is the pipeline a standard pipeline?
    environment : RuntimeEnvironment
        A description of the runtime environment, including engine versions,
        Docker images, compute resources, and benchmarks. If not provided,
        an attempt is made to determine it automatically.
    users : Sequence[User]
        Users associated with running the pipeline.

    Attributes
    ----------
    pipeline : Pipeline
        A pipeline to run.
    hyperparams : Sequence
        Values for free hyper-parameters of the pipeline. It should be a list, where each element corresponds
        to free hyper-parameters of the corresponding pipeline step. Not all free hyper-parameters have to be
        specified. Default values are used for those which are not. Optional.
    problem_description : Problem
        A parsed problem description in standard problem description schema.
    context : Context
        In which context to run pipelines, default is ``TESTING``.
    random_seed : int
        A random seed to use for every run. This control all randomness during the run.
    volumes_dir : str
        Path to a directory with static files required by primitives.
        In the standard directory structure (as obtained running ``python3 -m d3m.index download``).
    scratch_dir : str
        Path to a directory to store any temporary files needed during execution.
    is_standard_pipeline : bool
        Is the pipeline a standard pipeline?
    environment : RuntimeEnvironment
        A description of the runtime environment, including engine versions,
        Docker images, compute resources, and benchmarks. If not provided,
        an attempt is made to determine it automatically.
    users : Sequence[User]
        Users associated with running the pipeline.
    current_step : int
        Which step is currently being ran.
    phase : PipelineRunPhase
        Which phase are we currently running.
    pipeline_run : PipelineRun
        A current instance of pipeline run.
    return_values : Sequence[str]
        Which values should the runtime keep during a pipeline run, even after they are necessary.
    data_values : Dict[str, Any]
        Map between available data references and their values during the run.
    steps_state : List[Union[PrimitiveBase, Runtime]]
        Fitted state for each step of the pipeline.
    """

    def __init__(
        self, pipeline: pipeline_module.Pipeline, hyperparams: typing.Sequence = None, *,
        problem_description: problem.Problem = None, context: metadata_base.Context,
        random_seed: int = 0, volumes_dir: str = None, scratch_dir: str = None,
        is_standard_pipeline: bool = False, environment: pipeline_run_module.RuntimeEnvironment = None,
        users: typing.Sequence[pipeline_run_module.User] = None,
    ) -> None:
        self.pipeline = pipeline
        self.hyperparams = hyperparams
        self.problem_description = problem_description
        self.context = context
        self.random_seed = random_seed
        self.volumes_dir = volumes_dir
        self.scratch_dir = scratch_dir
        self.is_standard_pipeline = is_standard_pipeline
        self.users = users

        if environment is None:
            self.environment = pipeline_run_module.RuntimeEnvironment()
        else:
            self.environment = environment

        # Preliminary check.
        self.pipeline.check(allow_placeholders=False, standard_pipeline=self.is_standard_pipeline)

        if self.hyperparams is not None:
            self._check_hyperparams(self.pipeline, self.hyperparams)

        self.steps_state: typing.List[typing.Union[base.PrimitiveBase, Runtime, None]] = [None for step in self.pipeline.steps]

        self._previous_pipeline_run_id: str = None

        self._initialize_run_state([], None, None)

    def _initialize_data_values(self) -> None:
        # TODO: Remove values from the "data_values" once they are not needed anymore to optimize memory use.
        self.data_values: typing.Dict[str, typing.Any] = {}

    def _clear_data_values(self) -> None:
        self.data_values = {}

    def _initialize_run_state(
        self, inputs: typing.Sequence[typing.Any],
        phase: typing.Optional[metadata_base.PipelineRunPhase],
        return_values: typing.Optional[typing.Sequence[str]],
    ) -> None:
        self.current_step = 0
        self.phase = phase

        if return_values is None:
            self.return_values = self._get_all_outputs()
        else:
            # We sort "return_values" to have deterministic order.
            self.return_values = sorted(set(return_values))

        self._initialize_data_values()

        for i, input_value in enumerate(inputs):
            if isinstance(input_value, container.Dataset):
                input_value = self._mark_columns(input_value)
            else:
                # All standard pipeline inputs should be Datasets.
                assert not self.is_standard_pipeline

            self.data_values['inputs.{i}'.format(i=i)] = input_value

        self._initialize_base_temporary_directory()

        self._initialize_pipeline_run()

    def _get_all_outputs(self) -> typing.Sequence[str]:
        return ['outputs.{i}'.format(i=i) for i, output_description in enumerate(self.pipeline.outputs)]

    def _clear_run_state(self) -> None:
        """
        After a pipeline run, we clear state which was necessary while pipeline was running, but it is not needed anymore.
        """

        # We keep "steps_state" so that we can produce.

        self.current_step = 0
        self.phase = None
        self.return_values = None

        self._clear_data_values()
        self._clear_base_temporary_directory()
        self._clear_pipeline_run()

    def _check_hyperparams(self, pipeline: pipeline_module.Pipeline, hyperparams: typing.Sequence) -> None:
        """
        Check provided values for free hyper-parameters.
        """

        if not utils.is_sequence(hyperparams):
            raise exceptions.InvalidArgumentTypeError("Hyper-parameter values for the pipeline '{pipeline_id}' is not a sequence.".format(
                pipeline_id=pipeline.id,
            ))

        if len(hyperparams) != len(pipeline.steps):
            raise exceptions.InvalidArgumentValueError(
                "Hyper-parameter values for the pipeline '{pipeline_id}' do not match the number of steps in the pipeline: {hyperparams_steps} vs. {pipeline_steps}".format(
                    pipeline_id=pipeline.id,
                    hyperparams_steps=len(hyperparams),
                    pipeline_steps=len(pipeline.steps),
                ),
            )

        for step_index, (hyperparams_for_step, step) in enumerate(zip(hyperparams, pipeline.steps)):
            # Placeholder step is not really allowed, but we have it here for completeness.
            # Its "get_free_hyperparams" returns an empty list.
            if isinstance(step, pipeline_module.PlaceholderStep):
                if not utils.is_sequence(hyperparams_for_step):
                    raise exceptions.InvalidArgumentTypeError("Hyper-parameter values for placeholder step {step_index} of pipeline '{pipeline_id}' is not a sequence.".format(
                        step_index=step_index,
                        pipeline_id=pipeline.id,
                    ))

            elif isinstance(step, pipeline_module.SubpipelineStep):
                self._check_hyperparams(step.pipeline, hyperparams_for_step)

            elif isinstance(step, pipeline_module.PrimitiveStep):
                if not isinstance(hyperparams_for_step, (dict, frozendict.frozendict)):
                    raise exceptions.InvalidArgumentTypeError("Hyper-parameter values for primitive step {step_index} of pipeline '{pipeline_id}' is not a dict.".format(
                        step_index=step_index,
                        pipeline_id=pipeline.id,
                    ))

                hyperparams_for_step_keys = set(hyperparams_for_step.keys())
                free_hyperparams_keys = set(step.get_free_hyperparams().keys())
                all_hyperparams_keys = set(step.get_all_hyperparams().keys())

                if hyperparams_for_step_keys - all_hyperparams_keys:
                    raise exceptions.InvalidArgumentValueError(
                        "Hyper-parameter values for primitive step {step_index} of pipeline '{pipeline_id}' contain values for non-existent hyper-parameters: {hyperparams}".format(
                            step_index=step_index,
                            pipeline_id=pipeline.id,
                            hyperparams=sorted(hyperparams_for_step_keys - all_hyperparams_keys),
                        ),
                    )
                elif hyperparams_for_step_keys - free_hyperparams_keys:
                    raise exceptions.InvalidArgumentValueError(
                        "Hyper-parameter values for primitive step {step_index} of pipeline '{pipeline_id}' are overriding hyper-parameters fixed in the pipeline: {hyperparams}".format(
                            step_index=step_index,
                            pipeline_id=pipeline.id,
                            hyperparams=sorted(hyperparams_for_step_keys - free_hyperparams_keys),
                        ),
                    )

    def _get_pipeline_run_class(self) -> typing.Type[pipeline_run_module.PipelineRun]:
        return pipeline_run_module.PipelineRun

    def _initialize_pipeline_run(self) -> None:
        if self.phase is None:
            self.pipeline_run = None
            return

        self.pipeline_run = self._get_pipeline_run_class()(
            pipeline=self.pipeline,
            problem_description=self.problem_description,
            phase=self.phase,
            context=self.context,
            previous_pipeline_run_id=self._previous_pipeline_run_id,
            environment=self.environment,
            random_seed=self.random_seed,
            users=self.users
        )

        if self.is_standard_pipeline:
            input_values = []
            for i, input_value in sorted((int(data_reference.split('.')[1]), input_value) for data_reference, input_value in self.data_values.items() if data_reference.startswith('inputs.')):
                assert isinstance(input_value, container.Dataset), input_value
                input_values.append(input_value)

            for input_value in input_values:
                self.pipeline_run.add_input_dataset(input_value)

    def _clear_pipeline_run(self) -> None:
        self.pipeline_run = None

    def _initialize_base_temporary_directory(self) -> None:
        if self.phase is None:
            self._base_temporary_directory = None
            self._base_temporary_directory_path = None
            return

        self._base_temporary_directory = tempfile.TemporaryDirectory(dir=self.scratch_dir)
        self._base_temporary_directory_path = os.path.abspath(self._base_temporary_directory.name)

    def _clear_base_temporary_directory(self) -> None:
        if self._base_temporary_directory is not None:
            self._base_temporary_directory.cleanup()
            self._base_temporary_directory = None
            self._base_temporary_directory_path = None

    def _check_pipeline(self, inputs: typing.Sequence[typing.Any]) -> None:
        """
        Check with known inputs.
        """

        input_types = {}
        for i, input_value in enumerate(inputs):
            input_types['inputs.{i}'.format(i=i)] = type(input_value)

        self.pipeline.check(allow_placeholders=False, standard_pipeline=self.is_standard_pipeline, input_types=input_types)

    def _run_placeholder(self, step: pipeline_module.PlaceholderStep) -> None:
        raise exceptions.InvalidPipelineError("Step {step_index} of pipeline '{pipeline_id}' is a placeholder but there should be no placeholders.".format(
            step_index=self.current_step,
            pipeline_id=self.pipeline.id,
        ))

    # TODO: Make return type be equal to the current's class type, so that it adapts if this class is subclassed.
    def _create_subpipeline(self, pipeline: pipeline_module.Pipeline, hyperparams: typing.Optional[typing.Sequence]) -> 'Runtime':
        """
        Creates an instance of the subpipeline's runtime.
        """

        # We change the random seed in a deterministic way so that it does not matter in which order we run steps.
        # Subpipelines are generally not a standard pipeline.
        return type(self)(
            pipeline,
            hyperparams,
            # TODO: Should we pass "problem_description" as well, but make it so that it does not try to mark columns again?
            problem_description=None,
            context=self.context,
            random_seed=self.random_seed + self.current_step,
            volumes_dir=self.volumes_dir,
            scratch_dir=self.scratch_dir,
            is_standard_pipeline=False,
            environment=self.environment,
            users=self.users,
        )

    def _run_subpipeline(self, step: pipeline_module.SubpipelineStep) -> None:
        if step.pipeline is None:
            raise exceptions.InvalidPipelineError("Pipeline has not been resolved.")

        subpipeline_inputs: typing.List[typing.Any] = []
        for i, data_reference in enumerate(step.inputs):
            subpipeline_inputs.append(self.data_values[data_reference])

        if self.phase == metadata_base.PipelineRunPhase.FIT:
            if self.hyperparams is not None:
                hyperparams = self.hyperparams[self.current_step]

                # We checked this already in "_check_hyperparams".
                assert utils.is_sequence(hyperparams), hyperparams
            else:
                hyperparams = None

            subpipeline = self._create_subpipeline(step.pipeline, hyperparams)

            assert self.steps_state[self.current_step] is None
            self.steps_state[self.current_step] = subpipeline

        else:
            subpipeline = typing.cast(Runtime, self.steps_state[self.current_step])

            assert isinstance(subpipeline, Runtime), type(subpipeline)

        return_values_map = {}
        return_values = set()
        for i, output_id in enumerate(step.outputs):
            # "output_id" can be "None" if this output is not used and should be skipped.
            if output_id is not None:
                data_reference = 'outputs.{i}'.format(i=i)
                return_values.add(data_reference)
                return_values_map['steps.{i}.{output_id}'.format(i=step.index, output_id=output_id)] = data_reference

        step_reference_prefix = 'steps.{i}.'.format(i=step.index)
        for return_value in self.return_values:
            # We process recursive data references for this subpipeline.
            # We check that "return_value" is not in "return_values_map" because data
            # references of the format "steps.{i}.{output_id}" have "step_reference_prefix"
            # as a prefix but are not really a recursive data reference.
            # But all references of that format are already in "return_values_map".
            if return_value.startswith(step_reference_prefix) and return_value not in return_values_map:
                data_reference = return_value[len(step_reference_prefix):]
                # Data reference at this point should contain at least one dot, because all with the prefix
                # which do not contain a dot we filtered out by checking them against "return_values_map".
                assert '.' in data_reference, data_reference
                return_values.add(data_reference)
                return_values_map[return_value] = data_reference

        # We sort "return_values" to have deterministic order.
        result = subpipeline._run(subpipeline_inputs, self.phase, return_values=sorted(return_values))
        self.pipeline_run.add_subpipeline_step(result.pipeline_run)
        result.check_success()

        for step_data_reference, subpipeline_data_reference in return_values_map.items():
            self.data_values[step_data_reference] = result.values[subpipeline_data_reference]

    def _get_singleton_value(self, value: typing.Any, is_argument: bool, name: str) -> typing.Any:
        """
        A helper to extract a value from a singleton value (extracting a sole element of a
        container of length 1).
        """

        if len(value) != 1:
            if is_argument:
                raise exceptions.InvalidPipelineError(
                    "Argument '{argument_name}' of step {step_index} of pipeline '{pipeline_id}' is singleton data, but available data is not.".format(
                        argument_name=name,
                        step_index=self.current_step,
                        pipeline_id=self.pipeline.id,
                    ),
                )
            else:
                raise exceptions.InvalidPipelineError(
                    "Hyper-parameter '{hyperparameter_name}' of step {step_index} of pipeline '{pipeline_id}' is singleton data, but available data is not.".format(
                        hyperparameter_name=name,
                        step_index=self.current_step,
                        pipeline_id=self.pipeline.id,
                    ),
                )

        if isinstance(value, pandas.DataFrame):
            # Fetch the row as a list. This assures different columns can be of a different type.
            singleton_value = container.List([value.iloc[0, k] for k in range(len(value.columns))])
        else:
            singleton_value = value[0]

        if isinstance(singleton_value, types.Container):
            # TODO: Copy metadata from "value" to "singleton_value" as well.
            singleton_value.metadata = singleton_value.metadata.generate(singleton_value)

        return singleton_value

    def _prepare_primitive_arguments(self, step: pipeline_module.PrimitiveStep) -> typing.Dict[str, typing.Any]:
        arguments = {}
        for argument_name, argument_description in step.arguments.items():

            if argument_description['type'] == metadata_base.ArgumentType.DATA:
                argument_value = self.data_values[argument_description['data']]
                # We have to extract a singleton value out.
                argument_value = self._get_singleton_value(argument_value, True, argument_name)

            elif argument_description['type'] == metadata_base.ArgumentType.CONTAINER:
                if utils.is_sequence(argument_description['data']):
                    values = [self.data_values[data_reference] for data_reference in argument_description['data']]
                    # We have to create a container List.
                    argument_value = self._get_list_value(values)
                else:
                    argument_value = self.data_values[argument_description['data']]

            else:
                raise exceptions.UnexpectedValueError("Unknown argument type: {argument_type}".format(argument_type=argument_description['type']))

            arguments[argument_name] = argument_value

        return arguments

    def _get_list_value(self, values: typing.Sequence) -> container.List:
        """
        Creates a container List from ``values``. It reuses existing metadata in ``values``
        to create metadata of the container List.
        """

        container_list = container.List(values, {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.List,
            'dimension': {
                'length': len(values),
            },
        })

        for value_index, value in enumerate(values):
            container_list.metadata = value.metadata.copy_to(container_list.metadata, (), (value_index,))

        return container_list

    def _get_default_hyperparams(self, step: pipeline_module.PrimitiveStep) -> hyperparams_module.Hyperparams:
        return step.get_primitive_hyperparams().defaults()

    def _get_runtime_hyperparams(self, step: pipeline_module.PrimitiveStep) -> typing.Dict:
        if self.hyperparams is not None:
            runtime_hyperparams = self.hyperparams[self.current_step]

            # We checked this already in "_check_hyperparams".
            assert isinstance(runtime_hyperparams, (dict, frozendict.frozendict)), runtime_hyperparams
        else:
            runtime_hyperparams = {}

        return runtime_hyperparams

    def _get_pipeline_hyperparams(self, step: pipeline_module.PrimitiveStep) -> typing.Dict:
        pipeline_hyperparams = {}
        for hyperparameter_name, hyperparameter_description in step.hyperparams.items():
            if hyperparameter_description['type'] == metadata_base.ArgumentType.DATA:
                if utils.is_sequence(hyperparameter_description['data']):
                    pipeline_hyperparams[hyperparameter_name] = [
                        self._get_singleton_value(self.data_values[data_reference], False, hyperparameter_name)
                        for data_reference in hyperparameter_description['data']
                    ]
                else:
                    pipeline_hyperparams[hyperparameter_name] = self._get_singleton_value(self.data_values[hyperparameter_description['data']], False, hyperparameter_name)

            elif hyperparameter_description['type'] == metadata_base.ArgumentType.PRIMITIVE:
                if utils.is_sequence(hyperparameter_description['data']):
                    primitive_references = hyperparameter_description['data']
                else:
                    primitive_references = typing.cast(typing.Sequence, [hyperparameter_description['data']])

                primitives = []
                for primitive_reference in primitive_references:
                    primitive = self.steps_state[primitive_reference]

                    # It could point to a sub-pipeline and not a primitive.
                    if not isinstance(primitive, base.PrimitiveBase):
                        raise exceptions.InvalidPipelineError(
                            "Hyper-parameter '{hyperparameter_name}' of step {step_index} of pipeline '{pipeline_id}' does not point to a primitive step (step {primitive_reference}).".format(  # noqa
                                hyperparameter_name=hyperparameter_name,
                                step_index=self.current_step,
                                pipeline_id=self.pipeline.id,
                                primitive_reference=primitive_reference,
                            ),
                        )

                    # We do not yet make a copy of a primitive at this point.
                    # We do this later for all primitive values at once.
                    primitives.append(primitive)

                if utils.is_sequence(hyperparameter_description['data']):
                    pipeline_hyperparams[hyperparameter_name] = primitives
                else:
                    assert len(primitives) == 1

                    pipeline_hyperparams[hyperparameter_name] = primitives[0]  # type: ignore

            elif hyperparameter_description['type'] == metadata_base.ArgumentType.CONTAINER:
                pipeline_hyperparams[hyperparameter_name] = self.data_values[hyperparameter_description['data']]

            elif hyperparameter_description['type'] == metadata_base.ArgumentType.VALUE:
                pipeline_hyperparams[hyperparameter_name] = hyperparameter_description['data']

            else:
                raise exceptions.UnexpectedValueError("Unknown hyper-parameter type: {hyperparameter_type}".format(hyperparameter_type=hyperparameter_description['type']))

        return pipeline_hyperparams

    def _prepare_primitive_hyperparams(self, step: pipeline_module.PrimitiveStep) -> hyperparams_module.Hyperparams:
        default_hyperparams = self._get_default_hyperparams(step)
        pipeline_hyperparams = self._get_pipeline_hyperparams(step)
        runtime_hyperparams = self._get_runtime_hyperparams(step)

        # Pipeline hyper-parameters should be disjoint with runtime hyper-parameters.
        # We check this in "_check_hyperparams" call from the constructor.
        assert set(pipeline_hyperparams.keys()).isdisjoint(set(runtime_hyperparams.keys())), (pipeline_hyperparams, runtime_hyperparams)

        hyperparams = default_hyperparams.replace(pipeline_hyperparams).replace(runtime_hyperparams)

        self.pipeline_run.set_primitive_step_hyperparams(self.current_step, hyperparams, pipeline_hyperparams)

        # We have to handle all primitive values present in hyper-parameters.
        return self._handle_primitive_hyperparams(hyperparams, 0)

    def _filter_arguments(self, primitive_class: typing.Type[base.PrimitiveBase], method_name: str, arguments: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        """
        Primitive as a whole gets arguments for all its methods, so here we then filter out
        only those arguments expected by a given method.
        """

        method_arguments = primitive_class.metadata.query()['primitive_code'].get('instance_methods', {}).get(method_name, {}).get('arguments', [])

        filtered_arguments = {}
        for argument_name in method_arguments:
            if argument_name in arguments:
                filtered_arguments[argument_name] = arguments[argument_name]

        return filtered_arguments

    def _get_primitive_volumes(self, primitive_class: typing.Type[base.PrimitiveBase]) -> typing.Dict:
        volumes = {}
        for entry in primitive_class.metadata.get_volumes():
            if self.volumes_dir is None:
                raise exceptions.InvalidArgumentValueError(
                    "Primitive '{primitive_id}' of step {step_index} of pipeline '{pipeline_id}' requires static files (volumes) but volumes are not available.".format(
                        primitive_id=primitive_class.metadata.query()['id'],
                        step_index=self.current_step,
                        pipeline_id=self.pipeline.id,
                    ),
                )

            volume_path = os.path.join(self.volumes_dir, entry['file_digest'])
            if not os.path.exists(volume_path):
                raise exceptions.InvalidArgumentValueError(
                    "Primitive '{primitive_id}' of step {step_index} of pipeline '{pipeline_id}' requires static files (volume) but volume for key '{key}' is not available.".format(
                        primitive_id=primitive_class.metadata.query()['id'],
                        step_index=self.current_step,
                        pipeline_id=self.pipeline.id,
                        key=entry['key'],
                    ),
                )

            volumes[entry['key']] = volume_path

        return volumes

    def _get_primitive_temporary_directory(self, primitive_class: typing.Type[base.PrimitiveBase]) -> str:
        return tempfile.mkdtemp(dir=self._base_temporary_directory_path)

    def _create_primitive_arguments(self, primitive_class: typing.Type[base.PrimitiveBase], hyperparams: hyperparams_module.Hyperparams, random_seed_offset: int) -> typing.Dict:
        constructor_arguments = {
            'hyperparams': hyperparams,
            # We change the random seed in a deterministic way so that it does not matter in which order we run steps.
            'random_seed': self.random_seed + self.current_step + random_seed_offset,
            'volumes': self._get_primitive_volumes(primitive_class),
            'temporary_directory': self._get_primitive_temporary_directory(primitive_class),
        }

        filtered_arguments = self._filter_arguments(primitive_class, '__init__', constructor_arguments)

        return filtered_arguments

    def _create_primitive(self, primitive_class: typing.Type[base.PrimitiveBase], hyperparams: hyperparams_module.Hyperparams, random_seed_offset: int) -> base.PrimitiveBase:
        """
        Creates an instance of a non-pipeline primitive.

        Constructor call is not recorded in pipeline run.
        """

        arguments = self._create_primitive_arguments(primitive_class, hyperparams, random_seed_offset)

        return primitive_class(**arguments)

    def _clone_primitive(self, primitive: base.PrimitiveBase, random_seed_offset: int) -> base.PrimitiveBase:
        """
        Clone a primitive. It reuses hyper-parameters and params, but provides a
        potentially different random seed and other constructor arguments.

        We are creating a new instance and not a deep copy because primitive instance might have
        been created outside of the runtime and might not have valid constructor argument values.
        """

        # We have to handle all primitive values present in hyper-parameters.
        # They are all already an instance, but we have to make their copies.
        hyperparams = self._handle_primitive_hyperparams(primitive.hyperparams, random_seed_offset + 1)

        primitive_clone = self._create_primitive(type(primitive), hyperparams, random_seed_offset)

        primitive_clone.set_params(params=primitive.get_params())

        return primitive_clone

    def _create_pipeline_primitive(self, primitive_class: typing.Type[base.PrimitiveBase], hyperparams: hyperparams_module.Hyperparams) -> base.PrimitiveBase:
        """
        Creates an instance of a pipeline primitive.

        Constructor call is recorded in pipeline run.
        """

        arguments = self._create_primitive_arguments(primitive_class, hyperparams, 0)

        if 'random_seed' in arguments:
            self.pipeline_run.set_primitive_step_random_seed(self.current_step, arguments['random_seed'])

        return self._call_primitive_method(primitive_class, arguments)

    def _create_hyperparameter_primitive(self, primitive_class: typing.Type[base.PrimitiveBase], random_seed_offset: int) -> base.PrimitiveBase:
        """
        Creates an instance of the non-pipeline primitive with default hyper-parameters.
        """

        hyperparams_class = primitive_class.metadata.get_hyperparams()

        return self._create_primitive(primitive_class, hyperparams_class.defaults(), random_seed_offset)

    def _transform_primitive_hyperparameter(self, hyperparameter: hyperparams_module.Hyperparameter, value: typing.Any, index: int) -> typing.Any:
        value_is_type = utils.is_type(value)
        if value_is_type and issubclass(value, base.PrimitiveBase):
            return self._create_hyperparameter_primitive(value, index)
        elif not value_is_type and isinstance(value, base.PrimitiveBase):
            return self._clone_primitive(value, index)
        else:
            # Not a primitive instance or a primitive class, do not do anything.
            return value

    def _handle_primitive_hyperparams(self, hyperparams: base.Hyperparams, random_seed_offset: int) -> base.Hyperparams:
        """
        Handles a special case when the value is a primitive instance or a primitive class.
        In this case we have to make sure we create a new instance reusing its hyper-parameters,
        or create an instance from the class using default hyper-parameters.
        """

        return hyperparams.transform_value(hyperparams, self._transform_primitive_hyperparameter, random_seed_offset)

    def _run_primitive(self, step: pipeline_module.PrimitiveStep) -> None:
        if step.primitive is None:
            raise exceptions.InvalidPipelineError("Primitive has not been resolved.")

        self.pipeline_run.add_primitive_step(step)
        arguments = self._prepare_primitive_arguments(step)

        if self.phase == metadata_base.PipelineRunPhase.FIT:
            hyperparams = self._prepare_primitive_hyperparams(step)

            # We create a primitive just before it is being fitted for the first time. This assures that any primitives
            # it depends on through its hyper-parameters have already been fitted (because they are in prior steps).
            # Similarly, any pipeline-based value being passed to a hyper-parameter has already been computed.
            primitive = self._create_pipeline_primitive(step.primitive, hyperparams)

            assert self.steps_state[self.current_step] is None
            self.steps_state[self.current_step] = primitive

        else:
            primitive = typing.cast(base.PrimitiveBase, self.steps_state[self.current_step])

            assert isinstance(primitive, base.PrimitiveBase), type(primitive)

        if self.phase == metadata_base.PipelineRunPhase.FIT:
            fit_multi_produce_arguments = self._filter_arguments(step.primitive, 'fit_multi_produce', dict(arguments, produce_methods=step.outputs))

            # We fit and produce until fitting and producing finishes.
            # TODO: Support configuring limits on iterations/time.
            # TODO: Produce again only those produce methods which have not finished, currently we simply run all of them again.
            while True:
                multi_call_result = self._call_primitive_method(primitive.fit_multi_produce, fit_multi_produce_arguments)
                if multi_call_result.has_finished:
                    outputs = multi_call_result.values
                    break

        elif self.phase == metadata_base.PipelineRunPhase.PRODUCE:
            multi_produce_arguments = self._filter_arguments(step.primitive, 'multi_produce', dict(arguments, produce_methods=step.outputs))

            # We produce until producing finishes.
            # TODO: Support configuring limits on iterations/time.
            # TODO: Produce again only those produce methods which have not finished, currently we simply run all of them again.
            while True:
                multi_call_result = self._call_primitive_method(primitive.multi_produce, multi_produce_arguments)
                if multi_call_result.has_finished:
                    outputs = multi_call_result.values
                    break

        else:
            # TODO: Allow dispatch to a general method so that subclasses of this class can handle them if necessary.
            raise exceptions.UnexpectedValueError("Unknown phase: {phase}".format(phase=self.phase))

        for output_id in step.outputs:
            output_data_reference = 'steps.{i}.{output_id}'.format(i=step.index, output_id=output_id)

            if output_id in outputs:
                self.data_values[output_data_reference] = outputs[output_id]
            else:
                raise exceptions.InvalidReturnValueError("Missing declared output '{output_id}' in computed primitive's outputs.".format(output_id=output_id))

    def _call_primitive_method(self, method: typing.Callable, arguments: typing.Dict) -> typing.Any:
        """
        Calls a primitive method (or constructor). Records relevant information in pipeline run.

        Parameters
        ----------
        method : Callable
            Primitive's method or constructor to call.
        arguments : Dict
            Arguments to pass to the method.

        Returns
        -------
        Any
            The result of calling the method. It method is a constructor,
            returns an instance.
        """

        # A special case for the constructor.
        if inspect.isclass(method):
            method_name = '__init__'
        else:
            method_name = method.__name__

        pipeline_run_method_call_id = self.pipeline_run.add_method_call_to_primitive_step(self.current_step, method_name)

        callback = self.pipeline_run.get_method_call_logging_callback(pipeline_run_method_call_id)
        logging_handler = utils.CallbackHandler(callback)

        root = logging.getLogger()

        # TODO: All this redirection works in a single thread, what about multi-threaded or async?
        #       Reference engine is single threaded, but maybe a subclass would not be?
        # We redirect all stdout/stderr to logging.
        # with utils.redirect_to_logging():
        try:
            # Record all logging which happens during the call.
            root.addHandler(logging_handler)

            self.pipeline_run.method_call_started(pipeline_run_method_call_id)

            try:
                result = method(**arguments)
            except Exception as error:
                self.pipeline_run.method_call_failed(pipeline_run_method_call_id, traceback.format_exc())

                raise error

            self.pipeline_run.method_call_successful(pipeline_run_method_call_id)

        finally:
            root.removeHandler(logging_handler)

        self.pipeline_run.set_method_call_result_metadata(pipeline_run_method_call_id, result)

        return result

    def _run_step(self, step: pipeline_module.StepBase) -> None:
        if isinstance(step, pipeline_module.PlaceholderStep):
            self._run_placeholder(step)
        elif isinstance(step, pipeline_module.SubpipelineStep):
            self._run_subpipeline(step)
        elif isinstance(step, pipeline_module.PrimitiveStep):
            self._run_primitive(step)
        else:
            # TODO: Allow dispatch to a general method so that subclasses of this class can handle them if necessary.
            raise exceptions.UnexpectedValueError("Unknown step type: {step_type}".format(step_type=type(step)))

    def _do_run_step(self, step: pipeline_module.StepBase) -> None:
        self.pipeline_run.step_started(self.current_step)

        try:
            self._before_step_run()
            self._run_step(step)
            self._after_step_run()
        except Exception as error:
            self.pipeline_run.step_failed(self.current_step, traceback.format_exc())

            raise exceptions.StepFailedError(
                "Step {step_index} for pipeline {pipeline_id} failed.".format(
                    step_index=self.current_step, pipeline_id=self.pipeline.id,
                ),
            ) from error

        self.pipeline_run.step_successful(self.current_step)

    def _do_run(self) -> None:
        for step_index, step in enumerate(self.pipeline.steps):
            self.current_step = step_index

            self._do_run_step(step)

    def _run(
        self, inputs: typing.Sequence[typing.Any], phase: metadata_base.PipelineRunPhase,
        return_values: typing.Optional[typing.Sequence[str]]
    ) -> Result:
        self._check_pipeline(inputs)

        self._initialize_run_state(inputs, phase, return_values)

        self.pipeline_run.run_started()

        error: Exception = None
        try:
            self._do_run()
        except Exception as run_error:
            self.pipeline_run.run_failed(traceback.format_exc())

            error = run_error

        if error is None:
            self.pipeline_run.run_successful()

            self._populate_output_values()

            if self.is_standard_pipeline:
                self.pipeline_run.set_predictions(self.data_values['outputs.0'])

        values = self._get_return_values(error)

        pipeline_run = self.pipeline_run

        self._clear_run_state()

        # TODO: What if some internal exception happens before we set this which leaves runtime in a changed state.
        #       This means that state has changed, but we have not set previous pipeline run.
        #       So if another phase is called, it might even by accident succeed, but have invalid
        #       previous pipeline run set which does not explain the state of the runtime.
        #       Maybe we should make sure we always set this ID, even when not returning a pipeline
        #       run so that it can be at least visible that some pipeline run is missing in the sequence.
        self._previous_pipeline_run_id = pipeline_run.get_id()

        return Result(pipeline_run, values, error)

    def _get_return_values(self, error: typing.Optional[Exception]) -> typing.Dict:
        values = {}
        for name in self.return_values:
            try:
                values[name] = self.data_values[name]
            except KeyError as value_error:
                # We try to return whichever values we can, even in the case of an error.
                if error is None:
                    raise value_error

        return values

    def _before_step_run(self) -> None:
        pass

    def _after_step_run(self) -> None:
        self._delete_unnecessary_values()

    def _delete_unnecessary_values(self) -> None:
        values_needed = set()

        # Which values are explicitly required to be kept until the end?
        for value in self.return_values:
            values_needed.add(value)

        # Outputs need values from steps.
        for i, output_description in enumerate(self.pipeline.outputs):
            if 'outputs.{i}'.format(i=i) in self.return_values:
                values_needed.add(output_description['data'])

        # Future steps also need values.
        for step in self.pipeline.steps[self.current_step + 1:]:
            values_needed.update(step.get_input_data_references())

        # Pipeline run for a standard pipeline needs predictions.
        if self.is_standard_pipeline:
            values_needed.add(self.pipeline.outputs[0]['data'])

        # Delete any value which is not needed anymore.
        # We iterate over a list so that we can change dict while iterating.
        for data_reference in list(self.data_values.keys()):
            if data_reference not in values_needed:
                del self.data_values[data_reference]

    def fit(
        self, inputs: typing.Sequence[typing.Any], *, return_values: typing.Sequence[str] = None,
    ) -> Result:
        """
        Does a "fit" phase of the pipeline.

        Parameters
        ----------
        inputs : Sequence[Any]
            A list of inputs to the pipeline.
        return_values : Sequence[str]
            A list of data references of all output values of all steps to return.
            If ``None``, the output values of the whole pipeline are returned.

        Returns
        -------
        Result
            A result object with kept values, pipeline run description, and any exception.
        """

        return self._run(inputs, metadata_base.PipelineRunPhase.FIT, return_values)

    def produce(
        self, inputs: typing.Sequence[typing.Any], *, return_values: typing.Sequence[str] = None,
    ) -> Result:
        """
        Does a "produce" phase of the pipeline and returns outputs.

        Parameters
        ----------
        inputs : Sequence[Any]
            A list of inputs to the pipeline.
        return_values : Sequence[str]
            A list of data references of all output values of all steps to return.
            If ``None``, the output values of the whole pipeline are returned.

        Returns
        -------
        Result
            A result object with kept values, pipeline run description, and any exception.
        """

        return self._run(inputs, metadata_base.PipelineRunPhase.PRODUCE, return_values)

    def _populate_output_values(self) -> None:
        for i, output_description in enumerate(self.pipeline.outputs):
            # Outputs might not be available because they were not requested to be returned from the run.
            if output_description['data'] in self.data_values:
                self.data_values['outputs.{i}'.format(i=i)] = self.data_values[output_description['data']]

    # TODO: Warn if targets are in the problem description, but they have not been applied.
    # TODO: Warn if privileged data columns are in the problem description, but they have not been applied.
    def _mark_columns(self, dataset: container.Dataset) -> container.Dataset:
        if self.problem_description is None:
            return dataset

        dataset = dataset.copy()
        dataset_id = dataset.metadata.query(())['id']

        for problem_input in self.problem_description.get('inputs', []):
            # TODO: This is currently too strict because we have only one problem description for both train and test data.
            # if problem_input['dataset_id'] != dataset_id:
            #     continue

            for target in problem_input.get('targets', []):
                dataset.metadata = dataset.metadata.add_semantic_type(
                    (target['resource_id'], metadata_base.ALL_ELEMENTS, target['column_index']),
                    'https://metadata.datadrivendiscovery.org/types/Target',
                )
                dataset.metadata = dataset.metadata.add_semantic_type(
                    (target['resource_id'], metadata_base.ALL_ELEMENTS, target['column_index']),
                    'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                )
                # If column is marked as a target, it cannot be attribute as well.
                # This allows one to define in problem description otherwise attribute columns as targets.
                # See: https://gitlab.com/datadrivendiscovery/d3m/issues/265
                dataset.metadata = dataset.metadata.remove_semantic_type(
                    (target['resource_id'], metadata_base.ALL_ELEMENTS, target['column_index']),
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                )

            # TODO: Warn if privileged data columns are no set on attributes.
            for privileged_data in problem_input.get('privileged_data', []):
                dataset.metadata = dataset.metadata.add_semantic_type(
                    (privileged_data['resource_id'], metadata_base.ALL_ELEMENTS, privileged_data['column_index']),
                    'https://metadata.datadrivendiscovery.org/types/PrivilegedData',
                )

        return dataset


def _prepare_hyperparams(free_hyperparams: typing.Sequence, hyperparameter_values: typing.Dict) -> typing.Tuple[typing.Sequence, typing.Set[str]]:
    """
    Values in ``hyperparameter_values`` should be serialized as JSON, as obtained by JSON-serializing
    the output of hyper-parameter's ``value_to_json_structure`` method call.
    """

    hyperparams: typing.List[typing.Union[typing.Dict, typing.Sequence]] = []

    hyperparameter_values_used = set()

    for free_hyperparams_for_step in free_hyperparams:
        if isinstance(free_hyperparams_for_step, (dict, frozendict.frozendict)):
            values = {}
            for name, hyperparameter in free_hyperparams_for_step.items():
                if name in hyperparameter_values:
                    values[name] = hyperparameter.value_from_json_structure(json.loads(hyperparameter_values[name]))
                    hyperparameter_values_used.add(name)
            hyperparams.append(values)
        elif utils.is_sequence(free_hyperparams_for_step):
            step_hyperparams, step_hyperparameter_values_used = _prepare_hyperparams(free_hyperparams_for_step, hyperparameter_values)
            hyperparams.append(step_hyperparams)
            hyperparameter_values_used.update(step_hyperparameter_values_used)
        else:
            raise exceptions.UnexpectedValueError("Unknown hyper-parameters type: {hyperparams_type}".format(hyperparams_type=type(free_hyperparams_for_step)))

    return hyperparams, hyperparameter_values_used


# TODO: Add debug logging.
def fit(
    pipeline: pipeline_module.Pipeline, problem_description: problem.Problem, inputs: typing.Sequence[container.Dataset], *,
    context: metadata_base.Context, hyperparams: typing.Sequence = None, random_seed: int = 0, volumes_dir: str = None,
    scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
) -> typing.Tuple[typing.Optional[Runtime], typing.Optional[container.DataFrame], Result]:
    for input in inputs:
        if not isinstance(input, container.Dataset):
            raise TypeError("A standard pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(input),
            ))

    if len(pipeline.outputs) != 1:
        raise ValueError("A standard pipeline should have exactly one output, not {outputs}.".format(
            outputs=len(pipeline.outputs),
        ))

    runtime = Runtime(
        pipeline, hyperparams,
        problem_description=problem_description, context=context,
        random_seed=random_seed, volumes_dir=volumes_dir, scratch_dir=scratch_dir,
        is_standard_pipeline=True, environment=runtime_environment,
    )

    result = runtime.fit(inputs, return_values=['outputs.0'])
    if result.has_error():
        return None, None, result

    output = result.values['outputs.0']

    if not isinstance(output, container.DataFrame):
        raise TypeError("A standard pipeline's output should be of a container DataFrame type, not {output_type}.".format(
            output_type=type(output),
        ))

    return runtime, output, result


# TODO: Add debug logging.
def produce(
    fitted_pipeline: Runtime, test_inputs: typing.Sequence[container.Dataset],
) -> typing.Tuple[typing.Optional[container.DataFrame], Result]:
    for test_input in test_inputs:
        if not isinstance(test_input, container.Dataset):
            raise TypeError("A standard pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(test_input),
            ))

    # This is checked in "fit" already, but maybe somebody fitter a pipeline not through "fit".
    if len(fitted_pipeline.pipeline.outputs) != 1:
        raise ValueError("A standard pipeline should have exactly one output, not {outputs}.".format(
            outputs=len(fitted_pipeline.pipeline.outputs),
        ))

    result = fitted_pipeline.produce(test_inputs, return_values=['outputs.0'])
    if result.has_error():
        return None, result

    output = result.values['outputs.0']

    if not isinstance(output, container.DataFrame):
        raise TypeError("A standard pipeline's output should be of a container DataFrame type, not {output_type}.".format(
            output_type=type(output),
        ))

    return output, result


# TODO: Add debug logging.
def score(
    scoring_pipeline: pipeline_module.Pipeline, problem_description: problem.Problem, predictions: container.DataFrame,
    score_inputs: typing.Sequence[container.Dataset], metrics: typing.Sequence[typing.Dict], predictions_random_seed: int = None, *,
    context: metadata_base.Context, random_seed: int = 0, volumes_dir: str = None,
    scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
) -> typing.Tuple[typing.Optional[container.DataFrame], Result]:
    for score_input in score_inputs:
        if not isinstance(score_input, container.Dataset):
            raise TypeError("A scoring pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(score_input),
            ))

    if len(scoring_pipeline.outputs) != 1:
        raise ValueError("A scoring pipeline should have exactly one output, not {outputs}.".format(
            outputs=len(scoring_pipeline.outputs),
        ))

    if not metrics:
        raise exceptions.InvalidArgumentValueError("A list of metrics for scores to compute cannot be empty.")

    metrics_hyperparameter = []
    for metric in metrics:
        metric_hyperparameter = {'metric': metric['metric'].name, 'k': None, 'pos_label': None}
        metric_hyperparameter.update(metric.get('params', {}))
        metrics_hyperparameter.append(metric_hyperparameter)

    scoring_params = {
        # We have to JSON-serialize it because "_prepare_hyperparams" expects
        # all values to be JSON-serialized.
        'metrics': json.dumps(metrics_hyperparameter),
    }

    hyperparams, scoring_params_used = _prepare_hyperparams(scoring_pipeline.get_free_hyperparams(), scoring_params)

    scoring_params_keys_set = set(scoring_params.keys())
    if scoring_params_keys_set - scoring_params_used:
        logger.warning("Not all provided hyper-parameters for the scoring pipeline %(pipeline_id)s were used: %(unused_params)s", {
            'pipeline_id': scoring_pipeline.id,
            'unused_params': sorted(scoring_params_keys_set - scoring_params_used),
        })

    runtime = Runtime(
        scoring_pipeline, hyperparams,
        problem_description=problem_description, context=context,
        random_seed=random_seed, volumes_dir=volumes_dir, scratch_dir=scratch_dir,
        environment=runtime_environment,
    )

    inputs = [predictions] + list(score_inputs)  # type: ignore

    # Fit + produce on same data.
    result = runtime.fit(inputs, return_values=['outputs.0'])
    if result.has_error():
        return None, result

    output = result.values['outputs.0']

    if not isinstance(output, container.DataFrame):
        raise TypeError("A scoring pipeline's output should be of a container DataFrame type, not {output_type}.".format(
            output_type=type(output),
        ))

    if predictions_random_seed is not None:
        output = combine_random_seed(output, predictions_random_seed)

    return output, result


# TODO: Add debug logging.
def prepare_data(
    data_pipeline: pipeline_module.Pipeline, problem_description: problem.Problem, inputs: typing.Sequence[container.Dataset],
    data_params: typing.Dict[str, str], *, context: metadata_base.Context, random_seed: int = 0, volumes_dir: str = None,
    scratch_dir: str = None, runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
) -> typing.Tuple[typing.List, Result]:
    """
    Values in ``data_params`` should be serialized as JSON, as obtained by JSON-serializing
    the output of hyper-parameter's ``value_to_json_structure`` method call.
    """

    for input in inputs:
        if not isinstance(input, container.Dataset):
            raise TypeError("A data preparation pipeline's input should be of a container Dataset type, not {input_type}.".format(
                input_type=type(input),
            ))

    if len(data_pipeline.outputs) != 3:
        raise ValueError("A data preparation pipeline should have exactly three outputs, not {outputs}.".format(
            outputs=len(data_pipeline.outputs),
        ))

    if 'number_of_folds' in data_params:
        number_of_folds = int(data_params['number_of_folds'])
    else:
        # For now we assume other data preparation pipelines do only one fold. We should standardize
        # more hyper-parameters to gather how many folds have to be made (and not really folds, but
        # more how many input indices have to be passed to the pipeline).
        number_of_folds = 1

    data_hyperparams, data_params_used = _prepare_hyperparams(data_pipeline.get_free_hyperparams(), data_params)

    data_params_keys_set = set(data_params.keys())
    if data_params_keys_set - data_params_used:
        logger.warning("Not all provided hyper-parameters for the data preparation pipeline {pipeline_id} were used: {unused_params}".format(
            pipeline_id=data_pipeline.id,
            unused_params=sorted(data_params_keys_set - data_params_used),
        ))

    runtime = Runtime(
        data_pipeline, data_hyperparams,
        problem_description=problem_description, context=context,
        random_seed=random_seed, volumes_dir=volumes_dir,
        scratch_dir=scratch_dir, environment=runtime_environment,
    )

    # Fit + produce on same data. The inputs are the list of indices of folds
    # to generate and a dataset to split.
    result = runtime.fit([container.List(range(number_of_folds))] + list(inputs), return_values=['outputs.0', 'outputs.1', 'outputs.2'])  # type: ignore
    if result.has_error():
        return [], result

    outputs = [result.values['outputs.0'], result.values['outputs.1'], result.values['outputs.2']]

    for output in outputs:
        if not isinstance(output, container.List):
            raise TypeError("A data preparation pipeline's output should be of a container List type, not {input_type}.".format(
                input_type=type(output),
            ))
        if len(output) != number_of_folds:
            raise ValueError("A data preparation pipeline's output should contain {number_of_folds} datasets, not {length}.".format(
                number_of_folds=number_of_folds,
                length=len(output),
            ))

    return outputs, result


# TODO: Add debug logging.
def evaluate(
    pipeline: pipeline_module.Pipeline, data_pipeline: pipeline_module.Pipeline,
    scoring_pipeline: pipeline_module.Pipeline, problem_description: problem.Problem,
    inputs: typing.Sequence[container.Dataset], data_params: typing.Dict[str, str],
    metrics: typing.Sequence[typing.Dict], *, context: metadata_base.Context,
    hyperparams: typing.Sequence = None, random_seed: int = 0, data_random_seed: int = 0,
    scoring_random_seed: int = 0, volumes_dir: str = None, scratch_dir: str = None,
    runtime_environment: pipeline_run_module.RuntimeEnvironment = None,
) -> typing.Tuple[typing.List[container.DataFrame], MultiResult]:
    """
    Values in ``data_params`` should be serialized as JSON, as obtained by JSON-serializing
    the output of hyper-parameter's ``value_to_json_structure`` method call.
    """

    outputs, data_result = prepare_data(
        data_pipeline, problem_description, inputs, data_params,
        context=context, random_seed=data_random_seed, volumes_dir=volumes_dir,
        scratch_dir=scratch_dir, runtime_environment=runtime_environment,
    )
    if data_result.has_error():
        return [], MultiResult([data_result])

    fold_group_uuid = uuid.uuid4()

    all_scores: typing.List[container.DataFrame] = []
    all_results = MultiResult()
    for fold_index, (train_inputs, test_inputs, score_inputs) in enumerate(zip(*outputs)):
        fitted_pipeline, predictions, fit_result = fit(
            pipeline, problem_description, [train_inputs], context=context, hyperparams=hyperparams,
            random_seed=random_seed, volumes_dir=volumes_dir, scratch_dir=scratch_dir,
            runtime_environment=runtime_environment,
        )

        # Modifies "fit_result.pipeline_run" in-place.
        combine_pipeline_runs(
            fit_result.pipeline_run, data_pipeline_run=data_result.pipeline_run,
            fold_group_uuid=fold_group_uuid, fold_index=fold_index,
        )

        all_results.append(fit_result)
        if fit_result.has_error():
            return all_scores, all_results

        predictions, produce_result = produce(fitted_pipeline, [test_inputs])

        # Modifies "produce_result.pipeline_run" in-place.
        combine_pipeline_runs(
            produce_result.pipeline_run, data_pipeline_run=data_result.pipeline_run,
            fold_group_uuid=fold_group_uuid, fold_index=fold_index
        )

        all_results.append(produce_result)
        if produce_result.has_error():
            return all_scores, all_results

        scores, score_result = score(
            scoring_pipeline, problem_description, predictions, [score_inputs], metrics, random_seed,
            context=context, random_seed=scoring_random_seed, volumes_dir=volumes_dir,
            scratch_dir=scratch_dir, runtime_environment=runtime_environment,
        )

        # Modifies "produce_result.pipeline_run" in-place.
        combine_pipeline_runs(
            produce_result.pipeline_run, scoring_pipeline_run=score_result.pipeline_run,
        )

        # We modified "produce_result.pipeline_run" in-place and "produce_result.pipeline_run"
        # is already among "all_results", so we do not add it again.
        if score_result.has_error():
            return all_scores, all_results

        # Modifies "produce_result.pipeline_run" in-place.
        combine_pipeline_runs(
            produce_result.pipeline_run, metrics=metrics, scores=scores, problem_description=problem_description,
        )

        all_scores.append(scores)

    return all_scores, all_results


def get_pipeline(
    pipeline_path: str, *, strict_resolving: bool = False, strict_digest: bool = False,
    pipeline_search_paths: typing.Sequence[str] = None, respect_environment_variable: bool = True, load_all_primitives: bool = True,
    resolver_class: typing.Type[pipeline_module.Resolver] = pipeline_module.Resolver,
    pipeline_class: typing.Type[pipeline_module.Pipeline] = pipeline_module.Pipeline,
) -> pipeline_module.Pipeline:
    resolver = resolver_class(
        strict_resolving=strict_resolving, strict_digest=strict_digest, pipeline_search_paths=pipeline_search_paths,
        respect_environment_variable=respect_environment_variable, load_all_primitives=load_all_primitives,
    )

    if os.path.exists(pipeline_path):
        with open(pipeline_path, 'r', encoding='utf8') as pipeline_file:
            if pipeline_path.endswith('.yml'):
                return pipeline_class.from_yaml(pipeline_file, resolver=resolver, strict_digest=strict_digest)
            elif pipeline_path.endswith('.json'):
                return pipeline_class.from_json(pipeline_file, resolver=resolver, strict_digest=strict_digest)
            else:
                raise ValueError("Unknown file extension.")
    else:
        return resolver.get_pipeline({'id': pipeline_path})


def is_uri(uri: str) -> bool:
    """
    Test if a given string is an URI.

    Parameters
    ----------
    uri : str
        A potential URI to test.

    Returns
    -------
    bool
        ``True`` if string is an URI, ``False`` otherwise.
    """

    try:
        parsed_uri = url_parse.urlparse(uri)
    except Exception:
        return False

    return parsed_uri.scheme != ''


def get_dataset(dataset_uri: str, *, compute_digest: dataset_module.ComputeDigest = dataset_module.ComputeDigest.ONLY_IF_MISSING, strict_digest: bool = False) -> container.Dataset:
    if not is_uri(dataset_uri):
        dataset_uri = 'file://{dataset_doc_path}'.format(dataset_doc_path=os.path.abspath(dataset_uri))

    return container.Dataset.load(dataset_uri, compute_digest=compute_digest, strict_digest=strict_digest)


def get_problem(problem_uri: str) -> problem.Problem:
    if not is_uri(problem_uri):
        problem_uri = 'file://{problem_doc_path}'.format(problem_doc_path=os.path.abspath(problem_uri))

    return problem.Problem.load(problem_uri)


# TODO: Do not traverse the datasets directory every time.
def parse_meta(
    meta_file: typing.TextIO, datasets_dir: str, *, dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None, compute_digest: dataset_module.ComputeDigest = dataset_module.ComputeDigest.ONLY_IF_MISSING,
    strict_digest: bool = False, handle_score_split: bool = True,
) -> typing.Dict:
    if dataset_resolver is None:
        dataset_resolver = get_dataset
    if problem_resolver is None:
        problem_resolver = get_problem

    if datasets_dir is None:
        raise exceptions.InvalidArgumentValueError("Dataset directory has to be provided to resolve meta files.")

    meta = json.load(meta_file)

    datasets: typing.Dict[str, str] = {}
    problem_descriptions: typing.Dict[str, str] = {}

    for dirpath, dirnames, filenames in os.walk(datasets_dir, followlinks=True):
        dirpath = os.path.abspath(os.path.join(datasets_dir, dirpath))

        if 'datasetDoc.json' in filenames:
            # Do not traverse further (to not parse "datasetDoc.json" or "problemDoc.json" if they
            # exists in raw data filename).
            dirnames[:] = []

            dataset_path = os.path.join(dirpath, 'datasetDoc.json')

            try:
                with open(dataset_path, 'r', encoding='utf8') as dataset_file:
                    dataset_doc = json.load(dataset_file)

                dataset_id = dataset_doc['about']['datasetID']
                # Handle a special case for SCORE dataset splits (those which have "targets.csv" file).
                # They are the same as TEST dataset splits, but we present them differently, so that
                # SCORE dataset splits have targets as part of data. Because of this we also update
                # corresponding dataset ID.
                # See: https://gitlab.com/datadrivendiscovery/d3m/issues/176
                if handle_score_split and os.path.exists(os.path.join(dirpath, '..', 'targets.csv')) and dataset_id.endswith('_TEST'):
                    dataset_id = dataset_id[:-5] + '_SCORE'

                if dataset_id in datasets:
                    logger.warning(
                        "Duplicate dataset ID '%(dataset_id)s': '%(old_dataset)s' and '%(dataset)s'", {
                            'dataset_id': dataset_id,
                            'dataset': dataset_path,
                            'old_dataset': datasets[dataset_id],
                        },
                    )
                else:
                    datasets[dataset_id] = dataset_path

            except (ValueError, KeyError):
                logger.exception(
                    "Unable to read dataset '%(dataset)s'.", {
                        'dataset': dataset_path,
                    },
                )

        if 'problemDoc.json' in filenames:
            # We continue traversing further in this case.

            problem_path = os.path.join(dirpath, 'problemDoc.json')

            try:
                with open(problem_path, 'r', encoding='utf8') as problem_file:
                    problem_doc = json.load(problem_file)

                problem_id = problem_doc['about']['problemID']
                # Handle a special case for SCORE dataset splits (those which have "targets.csv" file).
                # They are the same as TEST dataset splits, but we present them differently, so that
                # SCORE dataset splits have targets as part of data. Because of this we also update
                # corresponding problem ID.
                # See: https://gitlab.com/datadrivendiscovery/d3m/issues/176
                if handle_score_split and os.path.exists(os.path.join(dirpath, '..', 'targets.csv')) and problem_id.endswith('_TEST'):
                    problem_id = problem_id[:-5] + '_SCORE'

                    # Also update dataset references.
                    for data in problem_doc.get('inputs', {}).get('data', []):
                        if data['datasetID'].endswith('_TEST'):
                            data['datasetID'] = data['datasetID'][:-5] + '_SCORE'

                if problem_id in problem_descriptions:
                    logger.warning(
                        "Duplicate problem ID '%(problem_id)s': '%(old_problem)s' and '%(problem)s'", {
                            'problem_id': problem_id,
                            'problem': problem_path,
                            'old_problem': problem_descriptions[problem_id],
                        },
                    )
                else:
                    problem_descriptions[problem_id] = problem_path

            except (ValueError, KeyError):
                logger.exception(
                    "Unable to read problem description '%(problem)s'.", {
                        'problem': problem_path,
                    },
                )

    return {
        'problem': problem_resolver(problem_descriptions[meta['problem']]),
        'full_inputs': [dataset_resolver(datasets[input_id], compute_digest=compute_digest, strict_digest=strict_digest) for input_id in meta['full_inputs']],
        'train_inputs': [dataset_resolver(datasets[input_id], compute_digest=compute_digest, strict_digest=strict_digest) for input_id in meta['train_inputs']],
        'test_inputs': [dataset_resolver(datasets[input_id], compute_digest=compute_digest, strict_digest=strict_digest) for input_id in meta['test_inputs']],
        'score_inputs': [dataset_resolver(datasets[input_id], compute_digest=compute_digest, strict_digest=strict_digest) for input_id in meta['score_inputs']],
    }


def combine_random_seed(scores: container.DataFrame, random_seed: int) -> container.DataFrame:
    random_seed_column = container.DataFrame({'randomSeed': [random_seed] * scores.shape[0]})
    # We add the new column at the end so that we do not have to do complicated changes to the metadata.
    output_scores = pandas.concat([scores, random_seed_column], axis=1)
    # There is one more column now, so we update metadata for it.
    output_scores.metadata = scores.metadata.update((metadata_base.ALL_ELEMENTS,), {
        'dimension': {
            'length': output_scores.shape[1],
        },
    })
    output_scores.metadata = output_scores.metadata.update_column(output_scores.shape[1] - 1, {
        'name': 'randomSeed',
        'structural_type': int,
    })

    return output_scores


def combine_folds(scores_list: typing.List[container.DataFrame]) -> container.DataFrame:
    # We combine multiple scores tables into one output table by adding a "fold" column.
    for fold, scores in enumerate(scores_list):
        fold_column = container.DataFrame({'fold': [fold] * scores.shape[0]})
        # We add the new column at the end so that we do not have to do complicated
        # changes to the metadata.
        scores_list[fold] = pandas.concat([scores, fold_column], axis=1)
        # There is one more column now, so we update metadata for it.
        scores_list[fold].metadata = scores.metadata.update((metadata_base.ALL_ELEMENTS,), {
            'dimension': {
                'length': scores_list[fold].shape[1],
            },
        })
        scores_list[fold].metadata = scores_list[fold].metadata.update_column(scores_list[fold].shape[1] - 1, {
            'name': 'fold',
            'structural_type': int,
        })

    scores = pandas.concat(scores_list, axis=0).reset_index(drop=True)
    # We reuse metadata from the first fold and update the number of rows which is now
    # combined across all folds.
    scores.metadata = scores_list[0].metadata.update((), {
        'dimension': {
            'length': scores.shape[0],
        },
    })

    return scores


def combine_pipeline_runs(
    standard_pipeline_run: pipeline_run_module.PipelineRun, *,
    data_pipeline_run: pipeline_run_module.PipelineRun = None, scoring_pipeline_run: pipeline_run_module.PipelineRun = None,
    metrics: typing.Sequence[typing.Dict] = None, scores: container.DataFrame = None,
    problem_description: problem.Problem = None,
    fold_group_uuid: uuid.UUID = None, fold_index: int = None,
) -> None:
    fold_args_provided = (item is None for item in (fold_group_uuid, fold_index))
    if any(fold_args_provided) and not all(fold_args_provided):
        raise exceptions.InvalidArgumentValueError("If any of 'fold_group_uuid' and 'fold_index' are provided, they must all be provided.")

    scores_args_provided = (item is None for item in (scores, metrics, problem_description))
    if any(scores_args_provided) and not all(scores_args_provided):
        raise exceptions.InvalidArgumentValueError("If any of 'scores', 'metrics', and 'problem_description' are provided, they must all be provided.")

    if data_pipeline_run is not None:
        standard_pipeline_run.set_data_preparation_pipeline_run(data_pipeline_run)

    if fold_group_uuid is not None:
        standard_pipeline_run.set_fold_group(fold_group_uuid, fold_index)

    if scoring_pipeline_run is not None:
        standard_pipeline_run.set_scoring_pipeline_run(scoring_pipeline_run)

    if scores is not None:
        standard_pipeline_run.set_scores(scores, metrics, problem_description)


@deprecate.function(message="use extended DataFrame.to_csv method instead")
def export_dataframe(dataframe: container.DataFrame, output_file: typing.TextIO = None) -> typing.Optional[str]:
    return dataframe.to_csv(output_file)


def get_metrics_from_list(metrics: typing.Sequence[str]) -> typing.Sequence[typing.Dict]:
    return [{'metric': problem.PerformanceMetric[metric]} for metric in metrics]


def get_metrics_from_problem_description(problem_description: problem.Problem) -> typing.Sequence[typing.Dict]:
    return problem_description['problem'].get('performance_metrics', [])


def _output_pipeline_runs(arguments: argparse.Namespace, pipeline_runs: typing.Sequence[pipeline_run_module.PipelineRun]) -> None:
    if not getattr(arguments, 'output_run', None):
        return

    first = True
    for pipeline_run in pipeline_runs:
        pipeline_run.to_yaml(arguments.output_run, appending=not first)
        first = False


def _fit(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None,
    meta_parser: typing.Callable = None, dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = get_dataset
    if problem_resolver is None:
        problem_resolver = get_problem

    context = metadata_base.Context[arguments.context]

    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    pipeline = pipeline_resolver(
        arguments.pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )

    if getattr(arguments, 'meta', None) is not None:
        meta = meta_parser(
            arguments.meta,
            getattr(arguments, 'datasets_dir', None),
            compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        problem_description = meta['problem']
        inputs = meta['train_inputs']
    else:
        problem_description = problem_resolver(arguments.problem)
        inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'inputs', [])
        ]

    fitted_pipeline, predictions, result = fit(
        pipeline, problem_description, inputs,
        context=context, random_seed=getattr(arguments, 'random_seed', 0),
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
    )

    _output_pipeline_runs(arguments, [result.pipeline_run])

    result.check_success()

    if getattr(arguments, 'save', None) is not None:
        pickle.dump(fitted_pipeline, arguments.save)

    if getattr(arguments, 'output', None) is not None:
        export_dataframe(predictions, arguments.output)


# We have "pipeline_resolver" as an argument (even if we are not using it
# in this function) so that the signature is the same for all handlers.
def _produce(arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None, dataset_resolver: typing.Callable = None) -> None:
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = get_dataset

    fitted_pipeline = pickle.load(arguments.fitted_pipeline)

    if getattr(arguments, 'meta', None) is not None:
        meta = meta_parser(
            arguments.meta,
            getattr(arguments, 'datasets_dir', None),
            compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        test_inputs = meta['test_inputs']
    else:
        test_inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'test_inputs', [])
        ]

    predictions, result = produce(fitted_pipeline, test_inputs)

    _output_pipeline_runs(arguments, [result.pipeline_run])

    result.check_success()

    if getattr(arguments, 'output', None) is not None:
        export_dataframe(predictions, arguments.output)


def _score(arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None, dataset_resolver: typing.Callable = None) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = get_dataset

    context = metadata_base.Context[arguments.context]

    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    fitted_pipeline = pickle.load(arguments.fitted_pipeline)
    scoring_pipeline = pipeline_resolver(
        arguments.scoring_pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )

    if getattr(arguments, 'meta', None) is not None:
        meta = meta_parser(
            arguments.meta,
            getattr(arguments, 'datasets_dir', None), compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        test_inputs = meta['test_inputs']
        score_inputs = meta['score_inputs']
    else:
        test_inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'test_inputs', [])
        ]
        score_inputs = [
            dataset_resolver(
                score_input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for score_input_uri in getattr(arguments, 'score_inputs', [])
        ]

    if getattr(arguments, 'metrics', None) is not None:
        metrics = get_metrics_from_list(arguments.metrics)
    else:
        metrics = get_metrics_from_problem_description(fitted_pipeline.problem_description)

    predictions, produce_result = produce(fitted_pipeline, test_inputs)

    if produce_result.has_error():
        _output_pipeline_runs(arguments, [produce_result.pipeline_run])
        produce_result.check_success()
        assert False

    if getattr(arguments, 'output', None) is not None:
        export_dataframe(predictions, arguments.output)

    scores, score_result = score(
        scoring_pipeline,
        fitted_pipeline.problem_description,
        predictions,
        score_inputs,
        metrics,
        fitted_pipeline.random_seed,
        context=context,
        random_seed=getattr(arguments, 'random_seed', 0),
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
    )

    # Modifies "produce_result.pipeline_run" in-place.
    combine_pipeline_runs(
        produce_result.pipeline_run, scoring_pipeline_run=score_result.pipeline_run,
    )

    if score_result.has_error():
        _output_pipeline_runs(arguments, [produce_result.pipeline_run])
        score_result.check_success()
        assert False

    # Modifies "produce_pipeline_run" in-place.
    combine_pipeline_runs(
        produce_result.pipeline_run, metrics=metrics, scores=scores, problem_description=fitted_pipeline.problem_description,
    )

    _output_pipeline_runs(arguments, [produce_result.pipeline_run])

    if getattr(arguments, 'scores', None) is not None:
        export_dataframe(scores, arguments.scores)


def _fit_produce(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None,
    meta_parser: typing.Callable = None, dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = get_dataset
    if problem_resolver is None:
        problem_resolver = get_problem

    context = metadata_base.Context[arguments.context]

    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    pipeline = pipeline_resolver(
        arguments.pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )

    if getattr(arguments, 'meta', None) is not None:
        meta = meta_parser(
            arguments.meta,
            getattr(arguments, 'datasets_dir', None),
            compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        problem_description = meta['problem']
        inputs = meta['train_inputs']
        test_inputs = meta['test_inputs']
    else:
        problem_description = problem_resolver(arguments.problem)
        inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'inputs', [])
        ]
        test_inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'test_inputs', [])
        ]

    fitted_pipeline, predictions, fit_result = fit(
        pipeline, problem_description, inputs, context=context,
        random_seed=getattr(arguments, 'random_seed', 0),
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
    )

    if fit_result.has_error():
        _output_pipeline_runs(arguments, [fit_result.pipeline_run])
        fit_result.check_success()
        assert False

    if getattr(arguments, 'save', None) is not None:
        pickle.dump(fitted_pipeline, arguments.save)

    predictions, produce_result = produce(fitted_pipeline, test_inputs)

    _output_pipeline_runs(arguments, [fit_result.pipeline_run, produce_result.pipeline_run])

    produce_result.check_success()

    if getattr(arguments, 'output', None) is not None:
        export_dataframe(predictions, arguments.output)


def _fit_score(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None,
    meta_parser: typing.Callable = None, dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = get_dataset
    if problem_resolver is None:
        problem_resolver = get_problem

    context = metadata_base.Context[arguments.context]

    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    pipeline = pipeline_resolver(
        arguments.pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )
    scoring_pipeline = pipeline_resolver(
        arguments.scoring_pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )

    if getattr(arguments, 'meta', None) is not None:
        meta = meta_parser(
            arguments.meta,
            getattr(arguments, 'datasets_dir', None), compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        problem_description = meta['problem']
        inputs = meta['train_inputs']
        test_inputs = meta['test_inputs']
        score_inputs = meta['score_inputs']
    else:
        problem_description = problem_resolver(arguments.problem)
        inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'inputs', [])
        ]
        test_inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'test_inputs', [])
        ]
        score_inputs = [
            dataset_resolver(
                score_input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for score_input_uri in getattr(arguments, 'score_inputs', [])
        ]

    if getattr(arguments, 'metrics', None) is not None:
        metrics = get_metrics_from_list(arguments.metrics)
    else:
        metrics = get_metrics_from_problem_description(problem_description)

    fitted_pipeline, predictions, fit_result = fit(
        pipeline, problem_description, inputs, context=context,
        random_seed=getattr(arguments, 'random_seed', 0),
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
    )

    if fit_result.has_error():
        _output_pipeline_runs(arguments, [fit_result.pipeline_run])
        fit_result.check_success()
        assert False

    if getattr(arguments, 'save', None) is not None:
        pickle.dump(fitted_pipeline, arguments.save)

    predictions, produce_result = produce(fitted_pipeline, test_inputs)

    if produce_result.has_error():
        _output_pipeline_runs(arguments, [fit_result.pipeline_run, produce_result.pipeline_run])
        produce_result.check_success()
        assert False

    if getattr(arguments, 'output', None) is not None:
        export_dataframe(predictions, arguments.output)

    scores, score_result = score(
        scoring_pipeline, problem_description, predictions, score_inputs, metrics, fitted_pipeline.random_seed,
        context=context, random_seed=getattr(arguments, 'scoring_random_seed', 0),
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
    )

    # Modifies "produce_result.pipeline_run" in-place.
    combine_pipeline_runs(
        produce_result.pipeline_run, scoring_pipeline_run=score_result.pipeline_run,
    )

    if score_result.has_error():
        _output_pipeline_runs(arguments, [fit_result.pipeline_run, produce_result.pipeline_run])
        score_result.check_success()
        assert False

    # Modifies "produce_result.pipeline_run" in-place.
    combine_pipeline_runs(
        produce_result.pipeline_run, metrics=metrics, scores=scores, problem_description=problem_description,
    )

    _output_pipeline_runs(arguments, [fit_result.pipeline_run, produce_result.pipeline_run])

    if getattr(arguments, 'scores', None) is not None:
        export_dataframe(scores, arguments.scores)


def _score_predictions(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None,
    meta_parser: typing.Callable = None, dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = get_dataset
    if problem_resolver is None:
        problem_resolver = get_problem

    context = metadata_base.Context[arguments.context]

    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    scoring_pipeline = pipeline_resolver(
        arguments.scoring_pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )

    if getattr(arguments, 'meta', None) is not None:
        meta = meta_parser(
            arguments.meta,
            getattr(arguments, 'datasets_dir', None), compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        problem_description = meta['problem']
        score_inputs = meta['score_inputs']
    else:
        problem_description = problem_resolver(arguments.problem)
        score_inputs = [
            dataset_resolver(
                score_input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for score_input_uri in getattr(arguments, 'score_inputs', [])
        ]

    if getattr(arguments, 'metrics', None) is not None:
        metrics = get_metrics_from_list(arguments.metrics)
    else:
        metrics = get_metrics_from_problem_description(problem_description)

    predictions_dataframe = pandas.read_csv(
        arguments.predictions,
        # We do not want to do any conversion of values at this point.
        # This should be done by primitives later on.
        dtype=str,
        # We always expect one row header.
        header=0,
        # We want empty strings and not NaNs.
        na_filter=False,
        encoding='utf8',
        low_memory=False,
        memory_map=True,
    )

    # Convert pandas DataFrame to container DataFrame.
    predictions = container.DataFrame(predictions_dataframe, generate_metadata=True)

    if getattr(arguments, 'output', None) is not None:
        export_dataframe(predictions, arguments.output)

    scores, score_result = score(
        scoring_pipeline, problem_description, predictions, score_inputs, metrics,
        getattr(arguments, 'predictions_random_seed', None),
        context=context, random_seed=getattr(arguments, 'scoring_random_seed', 0),
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
    )

    score_result.check_success()

    if getattr(arguments, 'scores', None) is not None:
        export_dataframe(scores, arguments.scores)


def _evaluate(
    arguments: argparse.Namespace, *, pipeline_resolver: typing.Callable = None,
    meta_parser: typing.Callable = None, dataset_resolver: typing.Callable = None,
    problem_resolver: typing.Callable = None,
) -> None:
    if pipeline_resolver is None:
        pipeline_resolver = get_pipeline
    if meta_parser is None:
        meta_parser = parse_meta
    if dataset_resolver is None:
        dataset_resolver = get_dataset
    if problem_resolver is None:
        problem_resolver = get_problem

    context = metadata_base.Context[arguments.context]

    runtime_environment = pipeline_run_module.RuntimeEnvironment(
        worker_id=getattr(arguments, 'worker_id', None),
    )

    pipeline = pipeline_resolver(
        arguments.pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )
    data_pipeline = pipeline_resolver(
        arguments.data_pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )
    scoring_pipeline = pipeline_resolver(
        arguments.scoring_pipeline,
        strict_resolving=getattr(arguments, 'strict_resolving', False),
        strict_digest=getattr(arguments, 'strict_digest', False),
        pipeline_search_paths=getattr(arguments, 'pipeline_search_paths', []),
    )

    if getattr(arguments, 'meta', None) is not None:
        meta = meta_parser(
            arguments.meta,
            getattr(arguments, 'datasets_dir', None), compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
            strict_digest=getattr(arguments, 'strict_digest', False),
        )
        problem_description = meta['problem']
        inputs = meta['full_inputs']
    else:
        problem_description = problem_resolver(arguments.problem)
        inputs = [
            dataset_resolver(
                input_uri,
                compute_digest=dataset_module.ComputeDigest[getattr(arguments, 'compute_digest', dataset_module.ComputeDigest.ONLY_IF_MISSING.name)],
                strict_digest=getattr(arguments, 'strict_digest', False),
            )
            for input_uri in getattr(arguments, 'inputs', [])
        ]

    if getattr(arguments, 'data_params', None) is not None:
        data_params = {name: value for name, value in arguments.data_params}
    else:
        data_params = {}

    if getattr(arguments, 'data_split_file', None) is not None:
        split_file = pandas.read_csv(
            arguments.data_split_file,
            # We do not want to do any conversion of values at this point.
            # This should be done by primitives later on.
            dtype=str,
            # We always expect one row header.
            header=0,
            # We want empty strings and not NaNs.
            na_filter=False,
            encoding='utf8',
            low_memory=False,
            memory_map=True,
        )

        # We use just the "d3mIndex" column and ignore multi-key indices.
        # This works for now because it seems that every current multi-key
        # dataset in fact has an unique value in "d3mIndex" alone.
        # See: https://gitlab.datadrivendiscovery.org/MIT-LL/d3m_data_supply/issues/117
        # Hyper-parameter value has to be JSON-serialized.
        data_params['primary_index_values'] = json.dumps(list(split_file.loc[split_file['type'] == 'TEST']['d3mIndex']))

    if getattr(arguments, 'metrics', None) is not None:
        metrics = get_metrics_from_list(arguments.metrics)
    else:
        metrics = get_metrics_from_problem_description(problem_description)

    scores_list, results_list = evaluate(
        pipeline, data_pipeline, scoring_pipeline, problem_description, inputs, data_params, metrics,
        context=context, random_seed=getattr(arguments, 'random_seed', 0),
        data_random_seed=getattr(arguments, 'data_random_seed', 0),
        scoring_random_seed=getattr(arguments, 'scoring_random_seed', 0),
        volumes_dir=getattr(arguments, 'volumes_dir', None),
        scratch_dir=getattr(arguments, 'scratch_dir', None),
        runtime_environment=runtime_environment,
    )

    _output_pipeline_runs(arguments, results_list.pipeline_runs)

    results_list.check_success()

    scores = combine_folds(scores_list)

    if getattr(arguments, 'scores', None) is not None:
        export_dataframe(scores, arguments.scores)


def handler(arguments: argparse.Namespace, parser: argparse.ArgumentParser, *,
            pipeline_resolver: typing.Callable = None, meta_parser: typing.Callable = None,
            dataset_resolver: typing.Callable = None, problem_resolver: typing.Callable = None) -> None:
    # Dynamically fetch which subparser was used.
    subparser = parser._subparsers._group_actions[0].choices[arguments.runtime_command]  # type: ignore

    if hasattr(arguments, 'meta'):
        # TODO: These arguments are required, but this is not visible from the usage line. These arguments are marked as optional there.
        manual_config = [('-r/--problem', 'problem'), ('-i/--input', 'inputs'), ('-t/--test-input', 'test_inputs'), ('-a/--score-input', 'score_inputs')]
        if any(hasattr(arguments, dest) and getattr(arguments, dest) is not None for (name, dest) in manual_config) and arguments.meta is not None:
            subparser.error("the following arguments cannot be used together: {manual_arguments} and -m/--meta".format(
                manual_arguments=', '.join(name for (name, dest) in manual_config if hasattr(arguments, dest) and getattr(arguments, dest) is not None),
            ))
        elif any(hasattr(arguments, dest) and getattr(arguments, dest) is None for (name, dest) in manual_config) and arguments.meta is None:
            subparser.error("the following arguments are required: {manual_arguments} or -m/--meta".format(
               manual_arguments=', '.join(name for (name, dest) in manual_config if hasattr(arguments, dest)),
            ))

    # Call a handler for the command.
    arguments.runtime_handler(arguments, pipeline_resolver=pipeline_resolver, meta_parser=meta_parser, dataset_resolver=dataset_resolver, problem_resolver=problem_resolver)


def configure_parser(parser: argparse.ArgumentParser, *, skip_arguments: typing.Tuple = ()) -> None:
    if 'random_seed' not in skip_arguments:
        parser.add_argument(
            '-n', '--random-seed', type=int, default=0, action='store', metavar='SEED',
            help="random seed to use",
        )
    if 'context' not in skip_arguments:
        parser.add_argument(
            '-x', '--context', choices=[context.name for context in metadata_base.Context], default=metadata_base.Context.TESTING.name, action='store',
            help="in which context to run pipelines, default is TESTING",
        )
    if 'pipeline_search_paths' not in skip_arguments:
        parser.add_argument(
            '-p', '--pipelines-path', action='append', metavar='PATH', dest='pipeline_search_paths',
            help="path to a directory with pipelines to resolve from (<pipeline id>.json and <pipeline id>.yml), "
                 "can be specified multiple times, has priority over PIPELINES_PATH environment variable",
        )
    if 'volumes_dir' not in skip_arguments:
        parser.add_argument(
            '-v', '--volumes', action='store', dest='volumes_dir',
            help="path to a directory with static files required by primitives, in the standard directory structure (as obtained running \"python3 -m d3m.index download\")",
        )
    if 'datasets_dir' not in skip_arguments:
        parser.add_argument(
            '-d', '--datasets', action='store', dest='datasets_dir',
            help="path to a directory with datasets (and problem descriptions) to resolve IDs in meta files",
        )
    if 'scratch_dir' not in skip_arguments:
        parser.add_argument(
            '-s', '--scratch', action='store', dest='scratch_dir',
            help="path to a directory to store any temporary files needed during execution",
        )
    if 'worker_id' not in skip_arguments:
        parser.add_argument(
            '--worker-id', action='store',
            help="globally unique identifier for the machine on which the runtime is running",
        )
    if 'compute_digest' not in skip_arguments:
        parser.add_argument(
            '--compute-digest', choices=[compute_digest.name for compute_digest in dataset_module.ComputeDigest], default=dataset_module.ComputeDigest.ONLY_IF_MISSING.name, action='store',
            help="when loading datasets, when to compute their digests, default is ONLY_IF_MISSING",
        )
    if 'strict_resolving' not in skip_arguments:
        parser.add_argument(
            '--strict-resolving', default=False, action='store_true',
            help="fail resolving if a resolved pipeline or primitive does not fully match specified reference",
        )
    if 'strict_digest' not in skip_arguments:
        parser.add_argument(
            '--strict-digest', default=False, action='store_true',
            help="when loading datasets or pipelines, if computed digest does not match the one provided in metadata, raise an exception?"
        )

    subparsers = parser.add_subparsers(dest='runtime_command', title='commands')
    subparsers.required = True  # type: ignore

    # TODO: Add command to compute "can_accept" over the pipeline.
    fit_parser = subparsers.add_parser(
        'fit', help="fit a pipeline",
        description="Fits a pipeline on train data, resulting in a fitted pipeline. Outputs also produced predictions during fitting on train data.",
    )
    produce_parser = subparsers.add_parser(
        'produce', help="produce using a fitted pipeline",
        description="Produce predictions on test data given a fitted pipeline.",
    )
    score_parser = subparsers.add_parser(
        'score', help="produce using a fitted pipeline and score results",
        description="Produce predictions on test data given a fitted pipeline and compute scores.",
    )
    fit_produce_parser = subparsers.add_parser(
        'fit-produce', help="fit a pipeline and then produce using it",
        description="Fit a pipeline on train data and produce predictions on test data.",
    )
    fit_score_parser = subparsers.add_parser(
        'fit-score', help="fit a pipeline, produce using it and score results",
        description="Fit a pipeline on train data, then produce predictions on test data and compute scores.",
    )
    score_predictions_parser = subparsers.add_parser(
        'score-predictions', help="score a predictions file",
        description="Compute scores given a file with predictions.",
    )
    evaluate_parser = subparsers.add_parser(
        'evaluate', help="evaluate a pipeline",
        description="Run pipeline multiple times using an evaluation approach and compute scores for each run.",
    )

    if 'pipeline' not in skip_arguments:
        fit_parser.add_argument(
            '-p', '--pipeline', action='store', required=True,
            help="path to a pipeline file (.json or .yml) or pipeline ID",
        )
    if 'problem' not in skip_arguments:
        fit_parser.add_argument(
            '-r', '--problem', action='store',
            help="path or URI to a problem description",
        )
    if 'inputs' not in skip_arguments:
        fit_parser.add_argument(
            '-i', '--input', action='append', metavar='INPUT', dest='inputs',
            help="path or URI of an input train dataset",
        )
    if 'meta' not in skip_arguments:
        fit_parser.add_argument(
            '-m', '--meta', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="path to a meta file with configuration",
        )
    if 'save' not in skip_arguments:
        fit_parser.add_argument(
            '-s', '--save', type=argparse.FileType('wb'), action='store',
            help="save fitted pipeline to a file",
        )
    if 'output' not in skip_arguments:
        fit_parser.add_argument(
            '-o', '--output', type=argparse.FileType('w', encoding='utf8'), default='-', action='store',
            help="save produced predictions during fitting to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        fit_parser.add_argument(
            '-O', '--output-run', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run document to a YAML file",
        )
    fit_parser.set_defaults(runtime_handler=_fit)

    if 'fitted_pipeline' not in skip_arguments:
        produce_parser.add_argument(
            '-f', '--fitted-pipeline', type=argparse.FileType('rb'), action='store', required=True,
            help="path to a saved fitted pipeline",
        )
    if 'test_inputs' not in skip_arguments:
        produce_parser.add_argument(
            '-t', '--test-input', action='append', metavar='INPUT', dest='test_inputs',
            help="path or URI of an input test dataset",
        )
    if 'meta' not in skip_arguments:
        produce_parser.add_argument(
            '-m', '--meta', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="path to a meta file with configuration",
        )
    if 'output' not in skip_arguments:
        produce_parser.add_argument(
            '-o', '--output', type=argparse.FileType('w', encoding='utf8'), default='-', action='store',
            help="save produced predictions to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        produce_parser.add_argument(
            '-O', '--output-run', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run document to a YAML file",
        )
    produce_parser.set_defaults(runtime_handler=_produce)

    if 'fitted_pipeline' not in skip_arguments:
        score_parser.add_argument(
            '-f', '--fitted-pipeline', type=argparse.FileType('rb'), action='store', required=True,
            help="path to a saved fitted pipeline",
        )
    if 'scoring_pipeline' not in skip_arguments:
        score_parser.add_argument(
            '-n', '--scoring-pipeline', action='store', required=True,
            help="path to a scoring pipeline file (.json or .yml) or pipeline ID",
        )
    if 'test_inputs' not in skip_arguments:
        score_parser.add_argument(
            '-t', '--test-input', action='append', metavar='INPUT', dest='test_inputs',
            help="path or URI of an input test dataset",
        )
    if 'score_inputs' not in skip_arguments:
        score_parser.add_argument(
            '-a', '--score-input', action='append', metavar='INPUT', dest='score_inputs',
            help="path or URI of an input score dataset",
        )
    if 'meta' not in skip_arguments:
        score_parser.add_argument(
            '-m', '--meta', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="path to a meta file with configuration",
        )
    if 'metrics' not in skip_arguments:
        score_parser.add_argument(
            '-e', '--metric', choices=[metric.name for metric in problem.PerformanceMetric], action='append', metavar='METRIC', dest='metrics',
            help="metric to use, using default parameters, can be specified multiple times, default from problem description",
        )
    if 'output' not in skip_arguments:
        score_parser.add_argument(
            '-o', '--output', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save produced predictions to a file",
        )
    if 'scores' not in skip_arguments:
        score_parser.add_argument(
            '-c', '--scores', type=argparse.FileType('w', encoding='utf8'), default='-', action='store',
            help="save scores to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        score_parser.add_argument(
            '-O', '--output-run', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run document to a YAML file",
        )
    score_parser.set_defaults(runtime_handler=_score)

    if 'pipeline' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-p', '--pipeline', action='store', required=True,
            help="path to a pipeline file (.json or .yml) or pipeline ID",
        )
    if 'problem' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-r', '--problem', action='store',
            help="path or URI to a problem description",
        )
    if 'inputs' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-i', '--input', action='append', metavar='INPUT', dest='inputs',
            help="path or URI of an input train dataset",
        )
    if 'test_inputs' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-t', '--test-input', action='append', metavar='INPUT', dest='test_inputs',
            help="path or URI of an input test dataset",
        )
    if 'meta' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-m', '--meta', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="path to a meta file with configuration",
        )
    if 'save' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-s', '--save', type=argparse.FileType('wb'), action='store',
            help="save fitted pipeline to a file",
        )
    if 'output' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-o', '--output', type=argparse.FileType('w', encoding='utf8'), default='-', action='store',
            help="save produced predictions to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        fit_produce_parser.add_argument(
            '-O', '--output-run', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run documents to a YAML file",
        )
    fit_produce_parser.set_defaults(runtime_handler=_fit_produce)

    if 'pipeline' not in skip_arguments:
        fit_score_parser.add_argument(
            '-p', '--pipeline', action='store', required=True,
            help="path to a pipeline file (.json or .yml) or pipeline ID",
        )
    if 'scoring_pipeline' not in skip_arguments:
        fit_score_parser.add_argument(
            '-n', '--scoring-pipeline', action='store', required=True,
            help="path to a scoring pipeline file (.json or .yml) or pipeline ID",
        )
    if 'problem' not in skip_arguments:
        fit_score_parser.add_argument(
            '-r', '--problem', action='store',
            help="path or URI to a problem description",
        )
    if 'inputs' not in skip_arguments:
        fit_score_parser.add_argument(
            '-i', '--input', action='append', metavar='INPUT', dest='inputs',
            help="path or URI of an input train dataset",
        )
    if 'test_inputs' not in skip_arguments:
        fit_score_parser.add_argument(
            '-t', '--test-input', action='append', metavar='INPUT', dest='test_inputs',
            help="path or URI of an input test dataset",
        )
    if 'score_inputs' not in skip_arguments:
        fit_score_parser.add_argument(
            '-a', '--score-input', action='append', metavar='INPUT', dest='score_inputs',
            help="path or URI of an input score dataset",
        )
    if 'meta' not in skip_arguments:
        fit_score_parser.add_argument(
            '-m', '--meta', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="path to a meta file with configuration",
        )
    if 'metrics' not in skip_arguments:
        fit_score_parser.add_argument(
            '-e', '--metric', choices=[metric.name for metric in problem.PerformanceMetric], action='append', metavar='METRIC', dest='metrics',
            help="metric to use, using default parameters, can be specified multiple times, default from problem description",
        )
    if 'save' not in skip_arguments:
        fit_score_parser.add_argument(
            '-s', '--save', type=argparse.FileType('wb'), action='store',
            help="save fitted pipeline to a file",
        )
    if 'output' not in skip_arguments:
        fit_score_parser.add_argument(
            '-o', '--output', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save produced predictions to a file",
        )
    if 'scores' not in skip_arguments:
        fit_score_parser.add_argument(
            '-c', '--scores', type=argparse.FileType('w', encoding='utf8'), default='-', action='store',
            help="save scores to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        fit_score_parser.add_argument(
            '-O', '--output-run', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run documents to a YAML file",
        )
    if 'scoring_random_seed' not in skip_arguments:
        fit_score_parser.add_argument(
            '--scoring-random-seed', type=int, action='store', default=0,
            help="random seed to use for scoring",
        )
    fit_score_parser.set_defaults(runtime_handler=_fit_score)

    if 'scoring_pipeline' not in skip_arguments:
        score_predictions_parser.add_argument(
            '-n', '--scoring-pipeline', action='store', required=True,
            help="path to a scoring pipeline file (.json or .yml) or pipeline ID",
        )
    if 'problem' not in skip_arguments:
        score_predictions_parser.add_argument(
            '-r', '--problem', action='store',
            help="path or URI to a problem description",
        )
    if 'predictions' not in skip_arguments:
        score_predictions_parser.add_argument(
            '-p', '--predictions', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="path to a predictions file",
        )
    if 'score_inputs' not in skip_arguments:
        score_predictions_parser.add_argument(
            '-a', '--score-input', action='append', metavar='INPUT', dest='score_inputs',
            help="path or URI of an input score dataset",
        )
    if 'meta' not in skip_arguments:
        score_predictions_parser.add_argument(
            '-m', '--meta', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="path to a meta file with configuration",
        )
    if 'metrics' not in skip_arguments:
        score_predictions_parser.add_argument(
            '-e', '--metric', choices=[metric.name for metric in problem.PerformanceMetric], action='append', metavar='METRIC', dest='metrics',
            help="metric to use, using default parameters, can be specified multiple times, default from problem description",
        )
    if 'scores' not in skip_arguments:
        score_predictions_parser.add_argument(
            '-c', '--scores', type=argparse.FileType('w', encoding='utf8'), default='-', action='store',
            help="save scores to a file, default stdout",
        )
    if 'scoring_random_seed' not in skip_arguments:
        score_predictions_parser.add_argument(
            '--scoring-random-seed', type=int, action='store', default=0,
            help="random seed to use for scoring",
        )
    if 'predictions_random_seed' not in skip_arguments:
        score_predictions_parser.add_argument(
            '--predictions-random-seed', type=int, action='store', default=None,
            help="random seed used for predictions",
        )
    score_predictions_parser.set_defaults(runtime_handler=_score_predictions)

    if 'pipeline' not in skip_arguments:
        evaluate_parser.add_argument(
            '-p', '--pipeline', action='store', required=True,
            help="path to a pipeline file (.json or .yml) or pipeline ID"
        )
    if 'data_pipeline' not in skip_arguments:
        evaluate_parser.add_argument(
            '-d', '--data-pipeline', action='store', required=True,
            help="path to a data preparation pipeline file (.json or .yml) or pipeline ID",
        )
    if 'scoring_pipeline' not in skip_arguments:
        evaluate_parser.add_argument(
            '-n', '--scoring-pipeline', action='store', required=True,
            help="path to a scoring pipeline file (.json or .yml) or pipeline ID",
        )
    if 'problem' not in skip_arguments:
        evaluate_parser.add_argument(
            '-r', '--problem', action='store',
            help="path or URI to a problem description",
        )
    if 'inputs' not in skip_arguments:
        evaluate_parser.add_argument(
            '-i', '--input', action='append', metavar='INPUT', dest='inputs',
            help="path or URI of an input full dataset",
        )
    if 'meta' not in skip_arguments:
        evaluate_parser.add_argument(
            '-m', '--meta', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="path to a meta file with configuration",
        )
    if 'data_params' not in skip_arguments:
        evaluate_parser.add_argument(
            '-y', '--data-param', nargs=2, action='append', metavar=('NAME', 'VALUE'), dest='data_params',
            help="hyper-parameter name and its value for data preparation pipeline, can be specified multiple times, value should be JSON-serialized",
        )
    if 'data_split_file' not in skip_arguments:
        evaluate_parser.add_argument(
            '--data-split-file', type=argparse.FileType('r', encoding='utf8'), action='store',
            help="reads the split file and populates \"primary_index_values\" hyper-parameter for data preparation pipeline with values from the \"d3mIndex\" column corresponding to the test data",
        )
    if 'metrics' not in skip_arguments:
        evaluate_parser.add_argument(
            '-e', '--metric', choices=[metric.name for metric in problem.PerformanceMetric], action='append', metavar='METRIC', dest='metrics',
            help="metric to use, using default parameters, can be specified multiple times, default from problem description",
        )
    if 'scores' not in skip_arguments:
        evaluate_parser.add_argument(
            '-c', '--scores', type=argparse.FileType('w', encoding='utf8'), default='-', action='store',
            help="save scores to a file, default stdout",
        )
    if 'output_run' not in skip_arguments:
        evaluate_parser.add_argument(
            '-O', '--output-run', type=argparse.FileType('w', encoding='utf8'), action='store',
            help="save pipeline run documents to a YAML file",
        )
    if 'data_random_seed' not in skip_arguments:
        evaluate_parser.add_argument(
            '--data-random-seed', type=int, action='store', default=0,
            help="random seed to use for data preparation",
        )
    if 'scoring_random_seed' not in skip_arguments:
        evaluate_parser.add_argument(
            '--scoring-random-seed', type=int, action='store', default=0,
            help="random seed to use for scoring",
        )
    evaluate_parser.set_defaults(runtime_handler=_evaluate)


def main(argv: typing.Sequence) -> None:
    logging.basicConfig()

    parser = argparse.ArgumentParser(description="Run D3M pipelines with default hyper-parameters.")
    configure_parser(parser)

    arguments = parser.parse_args(argv[1:])

    handler(arguments, parser)


if __name__ == '__main__':
    main(sys.argv)
