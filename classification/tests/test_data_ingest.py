import numpy as np

import pandas as pd

import configuration
import data_ingest


def test_data_ingest(
        raw_csv_data_with_semicolon_delimiter_path,
        data_set_read_settings_for_raw_csv_with_semicolon_delimiter
):
    data_set_ingest_settings, data_set_meta_data = data_set_read_settings_for_raw_csv_with_semicolon_delimiter

    data_set_ingest_result = data_ingest.read_data_set(
        raw_csv_data_with_semicolon_delimiter_path, data_set_ingest_settings, data_set_meta_data
    )

    expected_df = pd.DataFrame(
        {
            'col_1': np.array([2, 1, 4, 8, 32, 16, 64, 16], dtype=np.float32),
            'col_2': np.array([np.nan] * 8, dtype=np.float32),
            'label': pd.Categorical(
                values=['label_2', np.nan, 'label_1', 'label_1', 'label_2', 'label_4', 'label_4', 'label_1']
            ),
            'image': [
                'images/image_1.png',
                'images/image_1.png',
                'image_8.png',
                'image_4',
                '/image_32.jpg',
                'images/image_2',
                'images/image_2',
                'images/image_2'
            ],
            'cell_id': [0, 1, 0, 0, 0, 0, 1, 2],
            'col_3': np.array([0.5, 1.5, -0.25, 4444.0, 32.0, 0.5, -0.5, 0.75], dtype=np.float32),
            'col_4': np.array(
                [
                    configuration.PINF_REPRESENTATION,
                    configuration.NINF_REPRESENTATION,
                    configuration.PINF_REPRESENTATION,
                    configuration.NINF_REPRESENTATION,
                    configuration.PINF_REPRESENTATION,
                    configuration.NINF_REPRESENTATION,
                    configuration.PINF_REPRESENTATION,
                    configuration.NINF_REPRESENTATION,
                ],
                dtype=np.float32
            ),
        }
    )

    expected_predictor_columns = ['col_1', 'col_2', 'col_3', 'col_4']
    expected_label_column_name = 'label'
    expected_group_column_name = 'image'
    expected_cell_id_column_name = 'cell_id'

    pd.testing.assert_frame_equal(data_set_ingest_result.data_frame, expected_df)

    assert data_set_ingest_result.predictor_column_names == expected_predictor_columns
    assert data_set_ingest_result.label_column_name == expected_label_column_name
    assert data_set_ingest_result.group_column_name == expected_group_column_name
    assert data_set_ingest_result.cell_id_column_name == expected_cell_id_column_name
