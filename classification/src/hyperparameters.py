import typing


class HyperparametersMerger:
    def __init__(self):
        self._tunable_hyperparameters: list[str] = []
        self._fixed_hyperparameters: list[typing.Tuple[str, typing.Any]] = []

    def add_fixed_hyperparameter(self, name: str, value: typing.Any):
        self._fixed_hyperparameters.append((name, value))

    def add_tunable_hyperparameter(self, name: str):
        self._tunable_hyperparameters.append(name)

    def __call__(self, res_gp):
        best_parameters = {}
        for i, name in enumerate(self._tunable_hyperparameters):
            best_parameters[name] = res_gp.x[i]

        for name, value in self._fixed_hyperparameters:
            best_parameters[name] = value

        return best_parameters
