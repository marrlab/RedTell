import typing

from sklearn.impute import SimpleImputer
from skopt.space import Dimension

import configuration
import eda


class SimpleImputerPipelineStep:
    def __init__(self, data_set_eda: eda.DataSetEDA):
        self._data_set_eda: eda.DataSetEDA = data_set_eda

    @property
    def fixed_hyperparameters(self) -> dict[str, typing.Any]:
        fixed_hyperparameters = dict(
            missing_values=configuration.NAN_REPRESENTATION,
            strategy='median',
            add_indicator=False,
        )

        return fixed_hyperparameters

    @property
    def tunable_hyperparameters_sampling_space(self) -> list[Dimension]:
        sampling_space = []
        return sampling_space

    @property
    def estimator_cls(self):
        return SimpleImputer
