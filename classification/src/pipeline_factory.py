import typing

from sklearn.pipeline import Pipeline

import eda
from hyperparameters import HyperparametersMerger


def create_pipeline(pipeline_proto: list, data_set_eda: eda.DataSetEDA) -> \
        typing.Tuple[Pipeline, list, callable]:
    steps = []
    fixed_hyperparameters = {}
    tunable_hyperparameter_space = []
    merger = HyperparametersMerger()

    for name, pipeline_step_proto_cls in pipeline_proto:
        pipeline_step_proto = pipeline_step_proto_cls(data_set_eda=data_set_eda)

        steps.append((name, pipeline_step_proto.estimator_cls()))

        for fixed_hyperparameter, value in pipeline_step_proto.fixed_hyperparameters.items():
            hyperparameter_name = f'{name}__{fixed_hyperparameter}'
            fixed_hyperparameters[hyperparameter_name] = value
            merger.add_fixed_hyperparameter(hyperparameter_name, value)

        for dimension in pipeline_step_proto.tunable_hyperparameters_sampling_space:
            hyperparameter_name = f'{name}__{dimension.name}'
            dimension.name = hyperparameter_name
            tunable_hyperparameter_space.append(dimension)
            merger.add_tunable_hyperparameter(hyperparameter_name)

    pipeline = Pipeline(steps=steps)
    pipeline.set_params(**fixed_hyperparameters)

    return pipeline, tunable_hyperparameter_space, merger
