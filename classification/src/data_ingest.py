from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

import configuration

DTYPE = Union[str, np.dtype, type]


@dataclass
class DataSetMetaData:
    label_column_name: str
    group_column_name: str
    cell_id_column_name: str

    @property
    def non_predictor_column_names(self) -> set[str]:
        return {self.label_column_name, self.group_column_name, self.cell_id_column_name}


@dataclass(frozen=True)
class DataSetIngestSettings:
    delimiter: str
    predictor_column_dtype: DTYPE
    label_column_dtype: DTYPE
    cell_id_column_dtype: DTYPE
    group_column_dtype: DTYPE

    @classmethod
    def from_configuration(cls):
        return cls(
            delimiter=configuration.DELIMITER,
            predictor_column_dtype=configuration.PREDICTOR_COLUMN_DTYPE,
            label_column_dtype=configuration.LABEL_COLUMN_DTYPE,
            cell_id_column_dtype=configuration.CELL_ID_COLUMN_DTYPE,
            group_column_dtype=configuration.GROUP_COLUMN_DTYPE,
        )


@dataclass(frozen=True)
class DataIngestResult:
    data_frame: pd.DataFrame
    predictor_column_names: list[str]
    label_column_name: str
    group_column_name: str
    cell_id_column_name: str

    @property
    def all_target_labels(self) -> set:
        return set(self.subset_with_non_missing_label_values[self.label_column_name].unique())

    @property
    def missing_label_values_mask(self):
        return self.data_frame[self.label_column_name].isna()

    @property
    def subset_with_non_missing_label_values(self) -> pd.DataFrame:
        return self.data_frame[~self.missing_label_values_mask]
    @property
    def subset_with_missing_label_values(self) -> pd.DataFrame:
        return self.data_frame[self.missing_label_values_mask]


def _get_predictor_columns(df: pd.DataFrame, data_set_meta_data: DataSetMetaData) -> list[str]:
    return [
        column_name for column_name in df.columns if column_name not in data_set_meta_data.non_predictor_column_names
    ]


def _get_dtype_conversions(
        data_set_ingest_settings: DataSetIngestSettings,
        data_set_meta_data: DataSetMetaData,
        predictor_columns: list[str],
) -> dict[str, DTYPE]:
    types = {predictor_column: data_set_ingest_settings.predictor_column_dtype for predictor_column in
             predictor_columns}
    types[data_set_meta_data.label_column_name] = data_set_ingest_settings.label_column_dtype
    types[data_set_meta_data.cell_id_column_name] = data_set_ingest_settings.cell_id_column_dtype
    types[data_set_meta_data.group_column_name] = data_set_ingest_settings.group_column_dtype

    return types


def _clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data to clean.

    Returns
    -------
    cleaned_data : pd.DataFrame
        Cleaned data.
    """
    replacements = dict()

    for value in configuration.RAW_NAN_REPRESENTATIONS:
        replacements[value] = configuration.NAN_REPRESENTATION

    for value in configuration.RAW_PINF_REPRESENTATIONS:
        replacements[value] = configuration.PINF_REPRESENTATION

    for value in configuration.RAW_NINF_REPRESENTATIONS:
        replacements[value] = configuration.NINF_REPRESENTATION

    return df.replace(replacements)


def read_data_set(
        path: str, data_set_ingest_settings: DataSetIngestSettings, data_set_meta_data: DataSetMetaData
) -> DataIngestResult:
    """
    Read in raw data, clean it, and return together with associated metadata.

    Parameters
    ----------
    path : str
        Data set location.
    data_set_ingest_settings : DataSetIngestSettings
        Settings that determine how data should be read and cleaned.
    data_set_meta_data : DataSetMetaData
        Metadata associated with the data set.

    Returns
    -------
    data_set_ingest_result : DataIngestResult
        Cleaned data along with associated data set metadata.
    """
    df = pd.read_csv(path, delimiter=data_set_ingest_settings.delimiter, na_values=[None], keep_default_na=True)
    df = df.replace({"Echinocyte": "Stomatocyte"})
    predictor_column_names = _get_predictor_columns(df, data_set_meta_data)

    types = _get_dtype_conversions(data_set_ingest_settings, data_set_meta_data, predictor_column_names)

    df = df.astype(types)

    df = _clean_raw_data(df)

    return DataIngestResult(
        data_frame=df,
        predictor_column_names=predictor_column_names,
        label_column_name=data_set_meta_data.label_column_name,
        group_column_name=data_set_meta_data.group_column_name,
        cell_id_column_name=data_set_meta_data.cell_id_column_name,
    )
