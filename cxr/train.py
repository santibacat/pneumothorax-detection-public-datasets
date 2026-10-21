import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.saving import register_keras_serializable  # You need this import!


def keep_frontal_images(df):
    df = df.loc[(df.cxr_projection.isin(["AP", "PA"]))]
    return df


def split_and_balance_dataset(
    df, test_size=0.2, balance_ratio=2, random_state=42
):
    """
    Splits the dataset into train and test sets while ensuring:
    - The same patient does not appear in both sets.
    - Balancing the minority and majority classes based on a specified ratio.
    - Undersampling is performed per dataset.

    Parameters:
    df (pd.DataFrame): The input dataset with columns [patient_uuid, no_finding, pneumothorax, dataset].
    test_size (float): The proportion of data to be used for testing.
    balance_ratio (int): The ratio of majority class to minority class after undersampling.
    random_state (int): Random seed for reproducibility.

    Returns:
    dict: Dictionary containing 'train' and 'test' DataFrames.
    """
    np.random.seed(random_state)

    # Ensure that the same patient does not appear in both sets
    unique_patients = df["patient_uuid"].unique()
    train_patients, test_patients = train_test_split(
        unique_patients, test_size=test_size, random_state=random_state
    )

    train_df = df[df["patient_uuid"].isin(train_patients)].copy()
    test_df = df[df["patient_uuid"].isin(test_patients)].copy()

    def balance_dataset(sub_df, balance_ratio):
        """Performs undersampling to balance the dataset within each dataset."""

        pneumothorax_df = sub_df[sub_df["pneumothorax"] == 1]
        non_pneumothorax_df = sub_df[
            sub_df["pneumothorax"] == 0
        ]  # Includes both no_finding and other pathologies

        min_count = len(pneumothorax_df)
        max_count = min_count * balance_ratio

        # Undersample non_pneumothorax cases
        non_pneumothorax_df = non_pneumothorax_df.sample(
            n=min(max_count, len(non_pneumothorax_df)),
            random_state=random_state,
        )

        return (
            pd.concat([pneumothorax_df, non_pneumothorax_df])
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    # Perform balancing per dataset
    balanced_train_df = train_df.groupby("dataset", group_keys=False).apply(
        lambda x: balance_dataset(x, balance_ratio)
    )
    balanced_test_df = test_df.groupby("dataset", group_keys=False).apply(
        lambda x: balance_dataset(x, balance_ratio)
    )

    return {"train": balanced_train_df, "test": balanced_test_df}


def undersample_label(df, label_column, label_to_undersample, n_to_keep):
    sample = (
        df.loc[df[label_column] == label_to_undersample].shape[0] - n_to_keep
    )
    _ = df.loc[df[label_column] == label_to_undersample].sample(sample).index
    print(f"Undersampling label {label_to_undersample}")
    return df.drop(_, axis=0)


@register_keras_serializable()
def weighted_binary_crossentropy(class_weights):
    """
    Returns a weighted binary crossentropy loss function.

    Args:
        class_weights (list or array): A list or array of weights, one for each class/label.
        class_weights = [7.0, 1.0, 0.8]  # Pneumothorax, support devices, no finding

    Returns:
        loss function to use in model.compile
    """
    weights = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # Compute binary crossentropy per class
        bce = K.binary_crossentropy(y_true, y_pred)
        # Apply weights: shape (batch, num_classes)
        weighted_bce = bce * weights
        # Mean loss per sample
        return K.mean(weighted_bce, axis=-1)

    return loss


@register_keras_serializable()
def weighted_binary_crossentropy_per_class(
    class_weights, pos_weights=None, neg_weights=None
):
    """
    Returns a custom weighted binary crossentropy loss function supporting:
    - Class weights (importance of each class)
    - Positive sample bias (importance of positives)
    - Negative sample bias (importance of negatives)

    Args:
        class_weights (list/array): weight for each class output
        pos_weights (list/array): positive sample bias per class (default 1.0)
        neg_weights (list/array): negative sample bias per class (default 1.0)

        Eg:
        class_weights = [7.0, 1.0, 0.8]  # Pneumothorax, support devices, no finding
        pos_weights   = [2, 1.5, 1.0]  # Only boost pneumothorax positives
        neg_weights   = [0.8, 1.0, 1.0]  # No bias for negatives

    Returns:
        loss function to use in model.compile
    """
    class_weights = tf.constant(class_weights, dtype=tf.float32)
    if pos_weights is None:
        pos_weights = tf.ones_like(class_weights)
    else:
        pos_weights = tf.constant(pos_weights, dtype=tf.float32)

    if neg_weights is None:
        neg_weights = tf.ones_like(class_weights)
    else:
        neg_weights = tf.constant(neg_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        # Clip predictions to avoid log(0) errors
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Compute weighted binary crossentropy manually
        bce_per_class = -(
            pos_weights * y_true * K.log(y_pred)
            + neg_weights * (1 - y_true) * K.log(1 - y_pred)
        )

        # Apply class weights
        weighted_bce = bce_per_class * class_weights

        # Mean over classes
        return K.mean(
            weighted_bce, axis=-1
        )  # mean over classes, batch dimension handled by keras

    return loss


from sklearn.model_selection import StratifiedGroupKFold


def stratified_patient_split(
    df,
    label_cols,
    patient_col="patient_uuid",
    dataset_col="dataset",
    test_size=0.2,
    random_state=42,
):
    """
    Splits the dataframe into stratified train and test sets based on patients, preserving label distribution per dataset.

    Args:
        df (pd.DataFrame): Input dataframe.
        label_cols (list): List of label columns for stratification.
        patient_col (str): Column name of patient ID.
        dataset_col (str): Column name of dataset/clinic ID.
        test_size (float): Proportion for the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        train_df (pd.DataFrame), test_df (pd.DataFrame)
    """
    train_dfs = []
    test_dfs = []

    for dataset_name, dataset_df in df.groupby(dataset_col):
        # Group by patient and aggregate the labels
        patient_labels = (
            dataset_df.groupby(patient_col)[label_cols].max().reset_index()
        )

        # Create a "multi-label stratification label" by combining all labels
        # For simplicity, you can just take the label with highest prevalence (or hash labels)
        stratify_labels = (
            patient_labels[label_cols].astype(str).agg("".join, axis=1)
        )

        sgkf = StratifiedGroupKFold(
            n_splits=int(1 / test_size),
            shuffle=True,
            random_state=random_state,
        )

        for train_idx, test_idx in sgkf.split(
            patient_labels, stratify_labels, groups=patient_labels[patient_col]
        ):
            train_patients = patient_labels.iloc[train_idx][patient_col]
            test_patients = patient_labels.iloc[test_idx][patient_col]
            break  # Only one split needed

        train_df = dataset_df[dataset_df[patient_col].isin(train_patients)]
        test_df = dataset_df[dataset_df[patient_col].isin(test_patients)]

        train_dfs.append(train_df)
        test_dfs.append(test_df)

    full_train_df = pd.concat(train_dfs).reset_index(drop=True)
    full_test_df = pd.concat(test_dfs).reset_index(drop=True)

    return full_train_df, full_test_df


def fast_stratified_patient_split(
    df,
    label_cols,
    patient_col="patient_uuid",
    dataset_col="dataset",
    test_size=0.2,
    random_state=42,
):
    """
    FAST version of stratified patient split across multiple datasets.
    """

    # 1. Precompute: For each patient across all datasets
    patient_info = (
        df.groupby([dataset_col, patient_col])[label_cols].max().reset_index()
    )

    # 2. Create a stratification label
    stratify_labels = patient_info[label_cols].astype(str).agg("".join, axis=1)

    # 3. Prepare full index split
    sgkf = StratifiedGroupKFold(
        n_splits=int(1 / test_size), shuffle=True, random_state=random_state
    )

    train_patients = []
    test_patients = []

    for dataset_name, subset in patient_info.groupby(dataset_col):
        split = sgkf.split(
            subset,
            stratify_labels.loc[subset.index],
            groups=subset[patient_col],
        )

        for train_idx, test_idx in split:
            train_patients.append(subset.iloc[train_idx][patient_col])
            test_patients.append(subset.iloc[test_idx][patient_col])
            break  # Only need one split

    # 4. Flatten
    train_patients = pd.concat(train_patients)
    test_patients = pd.concat(test_patients)

    # 5. Select from full dataframe in one go
    full_train_df = df[df[patient_col].isin(train_patients)].reset_index(
        drop=True
    )
    full_test_df = df[df[patient_col].isin(test_patients)].reset_index(
        drop=True
    )

    return full_train_df, full_test_df


def ultra_fast_stratified_patient_split(
    df,
    label_cols,
    patient_col="patient_uuid",
    dataset_col="dataset",
    test_size=0.2,
    random_state=42,
):
    """
    Ultra-fast stratified patient split across multiple datasets.
    """

    # Precompute patient-level labels
    patient_info = (
        df.groupby([dataset_col, patient_col])[label_cols].max().reset_index()
    )

    # Numeric hashing instead of strings
    stratify_labels = patient_info[label_cols].dot(
        1 << np.arange(len(label_cols))
    )

    train_patients = []
    test_patients = []

    sgkf = StratifiedGroupKFold(
        n_splits=int(1 / test_size), shuffle=True, random_state=random_state
    )

    for dataset_name, subset in patient_info.groupby(dataset_col):
        split = sgkf.split(
            subset,
            stratify_labels.loc[subset.index],
            groups=subset[patient_col],
        )

        for train_idx, test_idx in split:
            train_patients.append(subset.iloc[train_idx][patient_col])
            test_patients.append(subset.iloc[test_idx][patient_col])
            break  # Only one split needed

    train_patients = pd.concat(train_patients)
    test_patients = pd.concat(test_patients)

    full_train_df = df[df[patient_col].isin(train_patients)].reset_index(
        drop=True
    )
    full_test_df = df[df[patient_col].isin(test_patients)].reset_index(
        drop=True
    )

    return full_train_df, full_test_df


def create_generators(
    df_train,
    df_test,
    x_col,
    y_col,
    augmentation=False,
    target_size=(224, 224),
    batch_size=16,
    seed=159,
    color_mode="rgb",
    *args,
    **kwargs,
):
    if augmentation:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            # rescale=1./255.,
            rotation_range=10,
            # zoom_range=[-1.02,1.02],
            horizontal_flip=True,
            vertical_flip=False,
            width_shift_range=[-0.03, 0.03],
            height_shift_range=[-0.03, 0.03],
            brightness_range=[0.8, 1.2],
            shear_range=3,
            fill_mode="nearest",
            preprocessing_function=None,
        )
    else:
        # train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    # test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, preprocessing_function = None)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=None
    )

    # Use dataframes to fill generators
    train_generator = train_datagen.flow_from_dataframe(
        df_train,
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        batch_size=batch_size,
        color_mode=color_mode,
        seed=seed,
        shuffle=True,
        class_mode="raw",
        validate_filenames=False,
    )

    valid_generator = test_datagen.flow_from_dataframe(
        df_test,
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        batch_size=batch_size,
        color_mode=color_mode,
        seed=seed,
        shuffle=False,
        class_mode="raw",
        validate_filenames=False,
    )

    test_generator = test_datagen.flow_from_dataframe(
        df_test,
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        batch_size=len(df_test),
        color_mode=color_mode,
        seed=seed,
        shuffle=False,
        class_mode="raw",
        validate_filenames=False,
    )

    return train_generator, valid_generator, test_generator
