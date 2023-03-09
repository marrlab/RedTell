import dataclasses
import random
import typing

import numpy as np

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, KFold

from data_ingest import DataIngestResult


class TrainingDataSet:
    def __init__(
            self,
            data_frame: pd.DataFrame,
            predictor_column_names: list[str],
            label_column_name: str,
            group_column_name: str,
    ):
        self._data_frame: pd.DataFrame = data_frame
        self._predictor_column_names: list[str] = predictor_column_names
        self._label_column_name: str = label_column_name
        self._group_column_name: str = group_column_name

    def sample(self, fraction: float = 1.0) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        fraction : float
            Sample size as fraction of the total number of observations.

        Returns
        -------
        x : np.ndarray
            Predictor values on the requested sample size.
        y : np.ndarray
            Label values on the requested sample size.
        groups : np.ndarray
            Group assignments of the sample rows.
        """
        assert 0.0 < fraction <= 1.0, 'sample must be between 0 and 1'

        df = self._data_frame.sample(frac=fraction, random_state=4444)

        return (
            df[self._predictor_column_names].values,
            df[self._label_column_name].values,
            df[self._group_column_name].values,
        )

    @property
    def total_num_observations(self):
        return len(self._data_frame)


class HoldoutDataSet:
    def __init__(
            self,
            data_frame: pd.DataFrame,
            predictor_column_names: list[str], label_column_name: str,
            group_column_name: str,
    ):
        self._data_frame: pd.DataFrame = data_frame
        self._predictor_column_names: list[str] = predictor_column_names
        self._label_column_name: str = label_column_name
        self._group_column_name: str = group_column_name

    @property
    def x(self) -> np.ndarray:
        return self._data_frame[self._predictor_column_names].values

    @property
    def y(self) -> np.ndarray:
        return self._data_frame[self._label_column_name].values

    @property
    def groups(self) -> np.ndarray:
        return self._data_frame[self._group_column_name].values

    def combine_with_predictions(self, y_pred) -> pd.DataFrame:
        df = self._data_frame
        df['y_pred'] = y_pred
        return df


class InferenceDataSet:
    def __init__(self, data_frame: pd.DataFrame, predictor_column_names: list[str]):
        self._data_frame: pd.DataFrame = data_frame
        self._predictor_column_names: list[str] = predictor_column_names

    @property
    def is_empty(self):
        return len(self._data_frame) == 0

    @property
    def x(self) -> np.ndarray:
        return self._data_frame[self._predictor_column_names].values

    def combine_with_predictions(self, y_pred) -> pd.DataFrame:
        df = self._data_frame
        df['y_pred'] = y_pred
        return df


def create_training_and_holdout_data_set(
        data_ingest_result: DataIngestResult,
        holdout_fraction: float = 0.33,
        partitioning_method: str = 'group',
) -> typing.Tuple[TrainingDataSet, HoldoutDataSet, InferenceDataSet]:
    df = data_ingest_result.subset_with_non_missing_label_values

    x = df[data_ingest_result.predictor_column_names].values
    y = df[data_ingest_result.label_column_name].values
    groups = df[data_ingest_result.group_column_name].values

    if partitioning_method == 'group':
        train_indices, holdout_indices = next(
            StratifiedGroupKFold(
                n_splits=int(round(1.0 / holdout_fraction)), shuffle=False,
            ).split(X=x, y=y, groups=groups)
        )
    elif partitioning_method == 'random':
        train_indices, holdout_indices = next(
            KFold(
                n_splits=int(round(1.0 / holdout_fraction)), shuffle=True,
            ).split(X=x, y=y)
        )

    else:
        raise ValueError(f'unknown partitioning method: {partitioning_method}')

    df_train = df.iloc[train_indices]
    df_holdout = df.iloc[holdout_indices]

    training_data_set = TrainingDataSet(
        data_frame=df_train,
        predictor_column_names=data_ingest_result.predictor_column_names,
        label_column_name=data_ingest_result.label_column_name,
        group_column_name=data_ingest_result.group_column_name,
    )

    holdout_data_set = HoldoutDataSet(
        data_frame=df_holdout,
        predictor_column_names=data_ingest_result.predictor_column_names,
        label_column_name=data_ingest_result.label_column_name,
        group_column_name=data_ingest_result.group_column_name,
    )

    inference_data_set = InferenceDataSet(
        data_frame=data_ingest_result.subset_with_missing_label_values,
        predictor_column_names=data_ingest_result.predictor_column_names,
    )

    return training_data_set, holdout_data_set, inference_data_set
