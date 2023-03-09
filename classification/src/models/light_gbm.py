import functools
import typing

from lightgbm import LGBMClassifier

from skopt.space.space import Dimension, Integer, Real

import eda
from pipeline_factory import create_pipeline


class LightGBMClassifierPipelineStep:
    def __init__(self, data_set_eda: eda.DataSetEDA):
        self._data_set_eda: eda.DataSetEDA = data_set_eda

    @property
    def fixed_hyperparameters(self) -> dict[str, typing.Any]:
        if self._data_set_eda.task_type == eda.TaskType.binary:
            objective = 'binary'
        elif self._data_set_eda.task_type == eda.TaskType.multiclass:
            objective = 'multiclass'
        else:
            raise ValueError(f'unsupported learning task: {self._data_set_eda.task_type.name}')

        fixed_hyperparameters = dict(
            boosting_type='goss',
            objective=objective,
            class_weight='balanced',
            random_state=0,
            n_jobs=-1,
        )

        return fixed_hyperparameters

    @property
    def tunable_hyperparameters_sampling_space(self) -> list[Dimension]:
        sampling_space = [
            Integer(low=32, high=100, prior='uniform', name='num_leaves'),
            Integer(low=1, high=10, prior='uniform', name='min_data_in_leaf'),
            Integer(low=1, high=10, prior='uniform', name='max_depth'),
            Integer(low=64, high=512, prior='uniform', name='max_bin'),
            Real(low=1e-4, high=1.0, prior='log-uniform', name='learning_rate'),
            Integer(low=100, high=5000, prior='log-uniform', name='num_iterations'),
            Real(low=0.25, high=1.0, prior='uniform', name='feature_fraction'),
            Real(low=1e-3, high=10.0, prior='log-uniform', name='lambda_l1'),
            Real(low=1e-3, high=10.0, prior='log-uniform', name='lambda_l2'),
        ]
        return sampling_space

    @property
    def estimator_cls(self):
        return LGBMClassifier


lightgbm_pipeline_proto = [
    ('classifier', LightGBMClassifierPipelineStep),
]

pipeline_factory = functools.partial(
    create_pipeline, pipeline_proto=lightgbm_pipeline_proto
)

name = 'LightGBM'
