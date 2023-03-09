import types

from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, StratifiedGroupKFold, GroupKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, recall_score, precision_score

from skopt import gp_minimize

from skopt.utils import use_named_args

import configuration
import data_sets
import eda


def create_objective(pipeline, hyperparameters, x, y, groups, cv):
    @use_named_args(hyperparameters)
    def objective(**hyperparameters):
        pipeline.set_params(**hyperparameters)

        return -cross_val_score(
            pipeline, x, y, cv=cv, groups=groups, n_jobs=-1, scoring='balanced_accuracy'
        ).mean()

    return objective


def fit_model(
        model: types.ModuleType,
        train_data_set: data_sets.TrainingDataSet,
        data_set_eda: eda.DataSetEDA,
        train_sample_fraction: float = 1.0,
        num_bayes_iterations: int = 10,
        partitioning_method: str = 'group',
):
    x_train, y_train, groups_train = train_data_set.sample(fraction=train_sample_fraction)

    if partitioning_method == 'group':
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    elif partitioning_method == 'random':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    else:
        raise ValueError(f'unsupported partitioning method: {partitioning_method}')

    pipeline, hyperparameters, hyperparameters_merger = \
        model.pipeline_factory(data_set_eda=data_set_eda)

    objective = create_objective(
        pipeline,
        hyperparameters,
        x_train,
        y_train,
        groups_train,
        cv,
    )

    res_gp = gp_minimize(
        objective,
        hyperparameters,
        n_calls=num_bayes_iterations,
        n_initial_points=10,
        random_state=0,
    )

    best_score = res_gp.fun
    best_parameters = hyperparameters_merger(res_gp)

    print(f'best score: {best_score}')
    print(f'best params: {best_parameters}')

    pipeline, _, _ = model.pipeline_factory(data_set_eda=data_set_eda)
    pipeline.set_params(**best_parameters)

    cv_scores = cross_val_score(
        pipeline,
        X=x_train,
        y=y_train,
        groups=groups_train,
        scoring='balanced_accuracy',
        cv=cv,
        n_jobs=-1,
    )

    pipeline, _, _ = model.pipeline_factory(data_set_eda=data_set_eda)
    pipeline.set_params(**best_parameters)

    pipeline.fit(X=x_train, y=y_train)

    train_score = balanced_accuracy_score(y_true=y_train, y_pred=pipeline.predict(x_train))

    return pipeline, cv_scores, train_score


def evaluate_model_on_holdout_set(
        model,
        holdout_data_set: data_sets.HoldoutDataSet,
):
    y_pred = model.predict(holdout_data_set.x)
    y_true = holdout_data_set.y

    balanced_accuracy_holdout = balanced_accuracy_score(y_true, y_pred)

    return balanced_accuracy_holdout


def train_model_and_generate_learning_curves(
        model: types.ModuleType,
        train_data_set: data_sets.TrainingDataSet,
        holdout_data_set: data_sets.HoldoutDataSet,
        learning_curve_sample_fractions: list[float],
        data_set_eda: eda.DataSetEDA,
        partitioning_method: str = 'group',
):
    num_observations_out = []

    train_scores_out = []
    cv_scores_out = []
    cv_scores_means_out = []

    holdout_scores_out = []

    balanced_accuracy_out = []
    balanced_accuracy_ova_out = defaultdict(list)

    accuracy_out = []
    accuracy_ova_out = defaultdict(list)

    f1_out = []
    f1_ova_out = defaultdict(list)

    recall_out = []
    recall_ova_out = defaultdict(list)

    precision_out = []
    precision_ova_out = defaultdict(list)

    for train_fraction in learning_curve_sample_fractions:
        num_observations = int(train_fraction * train_data_set.total_num_observations)

        print(f'training fraction: {train_fraction}; training size: {num_observations}')

        pipeline, cv_scores, train_score = fit_model(
            model,
            train_data_set,
            train_sample_fraction=train_fraction,
            num_bayes_iterations=configuration.NUM_BAYESIAN_OPTIMIZATION_ITERATIONS,
            data_set_eda=data_set_eda,
            partitioning_method=partitioning_method,
        )

        train_scores_out.append(train_score)

        holdout_scores = evaluate_model_on_holdout_set(pipeline, holdout_data_set)

        x_holdout = holdout_data_set.x
        y_holdout = holdout_data_set.y
        y_pred = pipeline.predict(x_holdout)

        balanced_accuracy = balanced_accuracy_score(y_holdout, y_pred)
        balanced_accuracy_out.append(balanced_accuracy)

        accuracy = accuracy_score(y_holdout, y_pred)
        accuracy_out.append(accuracy)

        f1 = f1_score(y_holdout, y_pred, average='macro')
        f1_out.append(f1)

        recall = recall_score(y_holdout, y_pred, average='macro')
        recall_out.append(recall)

        precision = precision_score(y_holdout, y_pred, average='macro')
        precision_out.append(precision)

        if data_set_eda.task_type == eda.TaskType.multiclass:

            labels = data_set_eda.unique_labels

            for label in labels:
                y_holdout_ova = (y_holdout == label).astype(np.float32)
                y_pred_ova = (y_pred == label).astype(np.float32)

                balanced_accuracy_ova = balanced_accuracy_score(y_holdout_ova, y_pred_ova)
                balanced_accuracy_ova_out[label].append(balanced_accuracy_ova)

                accuracy_ova = recall_score(y_holdout_ova, y_pred_ova)
                accuracy_ova_out[label].append(accuracy_ova)

                f1_ova = f1_score(y_holdout_ova, y_pred_ova, average='binary')
                f1_ova_out[label].append(f1_ova)

                recall_ova = recall_score(y_holdout_ova, y_pred_ova, average='binary')
                recall_ova_out[label].append(recall_ova)

                precision_ova = precision_score(y_holdout_ova, y_pred_ova, average='binary')
                precision_ova_out[label].append(precision_ova)

        num_observations_out.append(num_observations)

        cv_scores_out.append(cv_scores)

        cv_scores_means_out.append(cv_scores.mean())

        holdout_scores_out.append(holdout_scores)

    df_scores = pd.DataFrame(
        {
            'num_observations': num_observations_out,
            'balanced_accuracy': balanced_accuracy_out,
            'accuracy': accuracy_out,
            'f1_macro': f1_out,
            'recall': recall_out,
            'precision': precision_out,
        }
    )

    if data_set_eda.task_type == eda.TaskType.multiclass:

        labels = data_set_eda.unique_labels

        for label in labels:
            df_scores[f'{label}_balanced_accuracy'] = balanced_accuracy_ova_out[label]
            df_scores[f'{label}_accuracy'] = accuracy_ova_out[label]
            df_scores[f'{label}_f1'] = f1_ova_out[label]
            df_scores[f'{label}_recall'] = recall_ova_out[label]
            df_scores[f'{label}_precision'] = precision_ova_out[label]

    return (
        num_observations_out,
        train_scores_out,
        cv_scores_out,
        cv_scores_means_out,
        holdout_scores_out,
        balanced_accuracy_ova_out,
        df_scores,
    )
