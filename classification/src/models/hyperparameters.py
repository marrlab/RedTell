import abc

from typing import Any

from skopt.space.space import Dimension


class HyperParameters(abc.ABC):
    fixed_parameters: dict[str, Any] = None
    valid_parameters: set[str] = None

    @abc.abstractmethod
    def get_parameter_space(self) -> list[Dimension]:
        pass

    def merge_parameters(self, parameters: dict) -> dict:
        self._validate_parameters(parameters)
        return self.fixed_parameters | parameters

    def _validate_parameters(self, parameters: dict):
        assert set(parameters).issubset(self.valid_parameters), 'invalid parameter(-s)'

