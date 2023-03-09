import typing
import functools

from sklearn.tree import DecisionTreeClassifier
from skopt.space.space import Dimension, Integer, Real, Categorical

import eda
from models.imputation import SimpleImputerPipelineStep
from pipeline_factory import create_pipeline


class DecisionTreeClassifierPipelineStep:
    def __init__(self, data_set_eda: eda.DataSetEDA):
        self._data_set_eda: eda.DataSetEDA = data_set_eda

    @property
    def fixed_hyperparameters(self) -> dict[str, typing.Any]:
        fixed_hyperparameters = dict(
            random_state=0,
            class_weight='balanced',
        )

        return fixed_hyperparameters

    @property
    def tunable_hyperparameters_sampling_space(self) -> list[Dimension]:
        sampling_space = [
            Integer(low=1, high=50, prior='uniform', name='max_depth'),
            Integer(low=1, high=10, prior='uniform', name='min_samples_leaf'),
            Integer(low=2, high=10, prior='uniform', name='min_samples_split'),
            Real(low=0.2, high=1.0, prior='uniform', name='max_features'),
            Real(low=0.0, high=10.0, prior='uniform', name='ccp_alpha'),
            Categorical(categories=['gini', 'entropy', 'log_loss'], transform='identity', name='criterion'),
            Categorical(categories=['best', 'random'], transform='identity', name='splitter'),
        ]
        return sampling_space

    @property
    def estimator_cls(self):
        return DecisionTreeClassifier


random_forest_pipeline_proto = [
    ('imputer', SimpleImputerPipelineStep),
    ('classifier', DecisionTreeClassifierPipelineStep),
]

pipeline_factory = functools.partial(
    create_pipeline, pipeline_proto=random_forest_pipeline_proto
)

name = 'Decision Tree'
