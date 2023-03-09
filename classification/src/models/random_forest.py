import functools
import typing

from sklearn.ensemble import RandomForestClassifier

from skopt.space.space import Dimension, Integer, Real, Categorical

import eda
from models.imputation import SimpleImputerPipelineStep
from pipeline_factory import create_pipeline


class RandomForestClassifierPipelineStep:
    def __init__(self, data_set_eda: eda.DataSetEDA):
        self._data_set_eda: eda.DataSetEDA = data_set_eda

    @property
    def fixed_hyperparameters(self) -> dict[str, typing.Any]:
        fixed_hyperparameters = dict(
            n_jobs=-1,
            random_state=0,
            bootstrap=True,
            class_weight='balanced',
        )

        return fixed_hyperparameters

    @property
    def tunable_hyperparameters_sampling_space(self) -> list[Dimension]:
        sampling_space = [
            Integer(low=100, high=2_500, prior='log-uniform', name='n_estimators'),
            Integer(low=1, high=10, prior='uniform', name='max_depth'),
            Integer(low=1, high=10, prior='uniform', name='min_samples_leaf'),
            Real(low=0.2, high=1.0, prior='uniform', name='max_features'),
            Real(low=0.2, high=1.0, prior='uniform', name='max_samples'),
            Categorical(categories=['gini', 'entropy', 'log_loss'], transform='identity', name='criterion'),
        ]
        return sampling_space

    @property
    def estimator_cls(self):
        return RandomForestClassifier


random_forest_pipeline_proto = [
    ('imputer', SimpleImputerPipelineStep),
    ('classifier', RandomForestClassifierPipelineStep),
]

pipeline_factory = functools.partial(
    create_pipeline, pipeline_proto=random_forest_pipeline_proto
)


name = 'Random Forest'
