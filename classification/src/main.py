import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import configuration
import data_ingest
import data_sets
import eda

from bayesian_optimization_fitting import train_model_and_generate_learning_curves, fit_model

from models import light_gbm, random_forest, decision_tree


def main(
        path,
        label_column_name,
        cell_id_column_name,
        group_column_name=None,
        partitioning_method=None,
        output_folder='.',
):
    data_set_ingest_settings = data_ingest.DataSetIngestSettings.from_configuration()

    file_name = os.path.basename(path)

    data_set_meta_data = data_ingest.DataSetMetaData(
        label_column_name=label_column_name,
        group_column_name=group_column_name,
        cell_id_column_name=cell_id_column_name,
    )

    data_ingest_result = data_ingest.read_data_set(
        path, data_set_ingest_settings, data_set_meta_data
    )

    data_set_eda = eda.perform_eda(data_ingest_result)

    train_data_set, holdout_data_set, inference_data_set = data_sets.create_training_and_holdout_data_set(
        data_ingest_result, holdout_fraction=0.2559, partitioning_method=partitioning_method
    )

    num_training_samples = train_data_set.total_num_observations

    if num_training_samples < 100:
        raise ValueError(f'too few samples: {num_training_samples}')
    elif num_training_samples < 1_000:
        learning_curve_sample_fractions = [
           # 0.5, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
            1.0
        ]
    else:
        learning_curve_sample_fractions = [
            # 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.625, 0.75, 0.875, 1.0
            0.017, 0.033, 0.05, 0.083, # 0.1, 0.133, 0.167, 0.333 #, 0.666, 0.83, 1.0
        ]

    models = [decision_tree, random_forest, light_gbm]
    # models = [decision_tree]
    # models = [random_forest]

    for model in models:
        print(f'start fitting {model.name}')

        num_observations_out, \
        train_scores_out, \
        cv_scores_out, \
        cv_scores_means_out, \
        holdout_scores_out, \
        accuracy_ova_out,\
        df_scores = train_model_and_generate_learning_curves(
            model,
            train_data_set,
            holdout_data_set,
            learning_curve_sample_fractions=learning_curve_sample_fractions,
            data_set_eda=data_set_eda,
            partitioning_method=partitioning_method,
        )

        pipeline, _, _ = fit_model(
            model,
            train_data_set,
            train_sample_fraction=learning_curve_sample_fractions[-1],
            num_bayes_iterations=configuration.NUM_BAYESIAN_OPTIMIZATION_ITERATIONS,
            data_set_eda=data_set_eda,
            partitioning_method=partitioning_method,
        )

        _ = plt.figure(figsize=(16, 12))

        plt.plot(num_observations_out, train_scores_out, label='train')
        plt.plot(num_observations_out, cv_scores_means_out, label='cv')
        plt.plot(num_observations_out, holdout_scores_out, label='holdout')

        if data_set_eda.task_type == eda.TaskType.multiclass:
            labels = data_ingest_result.all_target_labels

            for label in labels:
                accuracy_ova = accuracy_ova_out[label]
                plt.plot(num_observations_out, accuracy_ova, label=f'holdout: {label}')

        plt.grid(True)

        plt.legend()

        out_dir = os.path.join(output_folder, file_name, model.name)

        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, 'learning_curve.png')

        plt.savefig(out_path)

        out_path = os.path.join(out_dir, 'learning_curves.csv')

        df_scores.to_csv(out_path, index=False)

        out_path = os.path.join(out_dir, 'holdout_predictions.csv')

        y_pred = pipeline.predict(holdout_data_set.x)
        df_holdout_predictions = holdout_data_set.combine_with_predictions(y_pred)
        df_holdout_predictions.to_csv(out_path, index=False)

        feature_names = data_ingest_result.predictor_column_names
        feature_importances = pipeline.named_steps['classifier'].feature_importances_

        max_importance = np.absolute(feature_importances).max()
        relative_importances = feature_importances / max_importance

        df_feature_importance = pd.DataFrame(
            {
                'feature': feature_names,
                'importance': relative_importances,
            }
        )

        df_feature_importance = df_feature_importance.sort_values(by='importance', axis=0, ascending=False)

        out_path = os.path.join(out_dir, 'feature_importance.csv')
        df_feature_importance.to_csv(out_path, index=False)

        fig, ax = plt.subplots(figsize=(16, 12))

        y_pos = np.arange(10)

        ax.barh(y_pos, df_feature_importance['importance'][:10], align='center')
        ax.set_yticks(y_pos, df_feature_importance['feature'][:10])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Feature importance (top 10 features)')

        plt.tight_layout()

        out_path = os.path.join(out_dir, 'feature_importance.png')

        plt.savefig(out_path)

        if not inference_data_set.is_empty:
            inference_predictions = pipeline.predict(inference_data_set.x)
            df_inference_predictions = inference_data_set.combine_with_predictions(inference_predictions)
            out_path = os.path.join(out_dir, 'inference_predictions.csv')
            df_inference_predictions.to_csv(out_path, index=False)
        else:
            print('skipping inference: no inference instances found in the data set')

    return 0


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='RedTell AutoML')

    parser.add_argument('-f', help='Input file path', required=True)
    parser.add_argument('-o', help='Output folder', required=False)
    parser.add_argument('--label', help='Label column name', required=True)
    parser.add_argument('--group', help='Group column name', required=False)
    parser.add_argument('--cell', help='Cell ID column name', required=True)
    parser.add_argument('-p', choices=['random', 'group'], help='Partitioning method', required=False)

    args = parser.parse_args()

    sys.exit(
        main(
            path=args.f,
            label_column_name=args.label,
            cell_id_column_name=args.cell,
            group_column_name=args.group,
            partitioning_method=args.p,
            output_folder=args.o,
        )
    )
