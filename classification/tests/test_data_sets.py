import numpy as np

import data_ingest

from data_sets import create_training_and_holdout_data_set


def test_training_holdout_split(
        raw_csv_data_with_semicolon_delimiter_no_missing_values_in_label_column,
        data_set_read_settings_for_raw_csv_data_with_semicolon_delimiter_no_missing_values_in_label_column,
):
    data_set_ingest_settings, data_set_meta_data = \
        data_set_read_settings_for_raw_csv_data_with_semicolon_delimiter_no_missing_values_in_label_column

    data_set_ingest_result = data_ingest.read_data_set(
        raw_csv_data_with_semicolon_delimiter_no_missing_values_in_label_column, data_set_ingest_settings,
        data_set_meta_data,
    )

    train_data_set, holdout_data_set = create_training_and_holdout_data_set(
        data_set_ingest_result, holdout_fraction=0.5
    )

    x_train, labels_train, groups_train = train_data_set.sample(fraction=1.0)

    expected_x_train = np.array(
        [
            [32.0, 32.0, 5.0],
            [64.0, -0.5, 7.0],
            [16.0, 0.75,  8.0],
            [16.0, 0.5, 6.0],
        ],
        dtype=np.float32
    )

    expected_labels_train = ['label_2', 'label_4', 'label_1', 'label_4']

    expected_groups_train = np.array(
        ['/image_32.jpg', 'images/image_2', 'images/image_2', 'images/image_2'], dtype=object
    )

    np.testing.assert_equal(x_train, expected_x_train)
    np.testing.assert_equal(list(labels_train), expected_labels_train)
    np.testing.assert_equal(groups_train, expected_groups_train)

    expected_x_holdout = np.array(
        [
            [2.0, 0.5, 1.0],
            [1.0, 1.5, 2.0],
            [4.0, -0.25, 3.0],
            [8.0, 4444.0, 4.0],
        ],
        dtype=np.float32,
    )

    expected_labels_holdout = ['label_2', 'label_2', 'label_1', 'label_1']

    expected_groups_holdout = ['images/image_1.png', 'images/image_1.png', 'image_8.png', 'image_4']

    np.testing.assert_equal(holdout_data_set.x, expected_x_holdout)
    np.testing.assert_equal(list(holdout_data_set.y), expected_labels_holdout)
    np.testing.assert_equal(holdout_data_set.groups, expected_groups_holdout)
