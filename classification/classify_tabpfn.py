import os
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
from tabpfn_extensions import interpretability

def split_annotated_data(
    data,
    train_fraction=0.8,
    random_state=None
):
    """
    Randomly split annotated rows into train and test data.

    Args:
        data (str | pandas.DataFrame): Either a data directory containing annotations.csv, a path to an annotations CSV, or an annotations DataFrame.
        train_fraction (float): Fraction of annotated rows assigned to train.
        random_state (int | None): Seed for reproducible random splits.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: The train and test DataFrames.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    if isinstance(data, pd.DataFrame):
        annotations = data.copy()
    else:
        annotations_path = data
        if os.path.isdir(annotations_path):
            annotations_path = os.path.join(annotations_path, "annotations.csv")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations file not found at {annotations_path}.")
        annotations = pd.read_csv(annotations_path, sep=None, engine="python")

    if "label" not in annotations.columns:
        raise ValueError("Annotated data must contain a 'label' column.")

    annotated_rows = annotations[
        annotations["label"].notna()
        & (annotations["label"].astype(str).str.strip() != "")
    ].copy()

    if annotated_rows.empty:
        raise ValueError("No annotated rows found. Please fill in the label column first.")

    shuffled = annotated_rows.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_size = int(len(shuffled) * train_fraction)

    train_data = shuffled.iloc[:train_size].reset_index(drop=True)
    test_data = shuffled.iloc[train_size:].reset_index(drop=True)

    return train_data, test_data

def merge_features_with_annotations(features_table, annotations):
    """
    Merge the features table with annotations based on cell_id.

    Args:
        features_table (pandas.DataFrame): DataFrame containing extracted features.
        annotations (pandas.DataFrame): DataFrame containing annotations.

    Returns:
        pandas.DataFrame: Merged DataFrame with features and corresponding labels.
    """
    if "cell_id" not in features_table.columns or "cell_id" not in annotations.columns:
        raise ValueError("Both features_table and annotations must contain a 'cell_id' column.")

    merged_data = pd.merge(
        features_table,
        annotations[["image", "cell_id", "label"]],
        on=["image", "cell_id"],
        how="left",
        validate="one_to_one",
    )
    return merged_data

def classify_tabpfn(img_dir):
    """
    Classifies cells using the TabPFN model based on extracted features.

    Args:
        img_dir (str): Directory containing the images and masks.
    """
    # Load the features table
    print("Loading features and annotations...")
    features_table_path = os.path.join(img_dir, "features_reddino.csv")
    if not os.path.exists(features_table_path):
        raise FileNotFoundError(f"Features table not found at {features_table_path}. Please run feature extraction first.")

    features_table = pd.read_csv(features_table_path)

    # load annotations
    annotations_path = os.path.join(img_dir, "annotations.csv")
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations file not found at {annotations_path}. Please run annotation generation first.")
    
    annotations = pd.read_csv(annotations_path, sep=";")

    print("Merging features with annotations...")
    data_table = merge_features_with_annotations(features_table, annotations)

    label_column = 'label'

    # split the data into test and train data
    print("Splitting data into train and test sets...")
    train_data, test_data = split_annotated_data(data_table, train_fraction=0.8, random_state=42)

    # Load the TabPFN model
    print("Loading TabPFN model...")
    model = TabPFNClassifier()

    # Prepare the feature data for classification
    feature_columns = [col for col in features_table.columns if col.startswith('feature_')]
    
    X_train = train_data[feature_columns].values
    y_train = train_data[label_column].values

    X_test = test_data[feature_columns].values
    y_test = test_data[label_column].values

    # Perform classification
    print("Training the model and making predictions...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Add predictions to the DataFrame
    results = test_data[['image', 'cell_id', 'label']]
    results['predicted_label'] = predictions

    # Save the updated DataFrame with predictions
    output_path = os.path.join(img_dir, "classified_cells.csv")
    results.to_csv(output_path, index=False)
    print(f"Classification completed. Results saved to {output_path}.")
