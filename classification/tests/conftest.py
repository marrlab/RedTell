import pathlib

import numpy as np

import pytest

import data_ingest


@pytest.fixture(scope='function')
def raw_csv_data_with_semicolon_delimiter_path():
    return pathlib.Path(__file__).parent / 'fixtures' / 'raw_csv_data_with_semicolon_delimiter.csv'


@pytest.fixture(scope='function')
def raw_csv_data_with_semicolon_delimiter_no_missing_values_in_label_column():
    return pathlib.Path(__file__).parent \
           / 'fixtures' \
           / 'raw_csv_data_with_semicolon_delimiter_no_missing_values_in_label_column.csv'


@pytest.fixture(scope='function')
def data_set_read_settings_for_raw_csv_with_semicolon_delimiter():
    data_set_ingest_settings = data_ingest.DataSetIngestSettings(
        delimiter=';',
        predictor_column_dtype=np.float32,
        label_column_dtype='category',
        cell_id_column_dtype=np.int64,
        group_column_dtype=str,
    )

    data_set_meta_data = data_ingest.DataSetMetaData(
        label_column_name='label',
        group_column_name='image',
        cell_id_column_name='cell_id',
    )

    return data_set_ingest_settings, data_set_meta_data


@pytest.fixture(scope='function')
def data_set_read_settings_for_raw_csv_data_with_semicolon_delimiter_no_missing_values_in_label_column():
    data_set_ingest_settings = data_ingest.DataSetIngestSettings(
        delimiter=';',
        predictor_column_dtype=np.float32,
        label_column_dtype='category',
        cell_id_column_dtype=np.int64,
        group_column_dtype=str,
    )

    data_set_meta_data = data_ingest.DataSetMetaData(
        label_column_name='label',
        group_column_name='image',
        cell_id_column_name='cell_id',
    )

    return data_set_ingest_settings, data_set_meta_data
