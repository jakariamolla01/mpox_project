"""
Dataset assumed in folder-per-class layout, e.g.:

    DATA_DIR/
        Monkeypox/*.jpg
        Chickenpox/*.jpg
        Measles/*.jpg
        ...

Builds a 5-fold split (StratifiedKFold) on file paths, then per-fold
ImageDataGenerator pipelines: heavy augmentation on train, none on val/test.
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_dataframe(data_dir):
    """Scans DATA_DIR/<class_name>/* and returns a DataFrame [filepath, label]."""
    rows = []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    for cls in classes:
        for fp in glob.glob(os.path.join(data_dir, cls, "*")):
            rows.append({"filepath": fp, "label": cls})
    df = pd.DataFrame(rows)
    print(df["label"].value_counts())
    return df, classes


def get_kfold_splits(df, n_splits=5, seed=108):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(df["filepath"], df["label"]))
    return splits  # list of (train_idx, test_idx)


def make_generators(df, train_idx, test_idx, classes, img_size, batch_size,
                     preprocess_fn=None, val_from_train=0.1, seed=108):
    """
    Returns train_gen, val_gen, test_gen for one fold.
    Augmentation matches the paper: flips, rotation, brightness/contrast, etc.
    Uses `preprocess_fn` (backbone-specific) for normalization instead of /255
    if provided -- otherwise falls back to rescale=1/255 as the paper describes.
    """
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    common_kwargs = dict(preprocessing_function=preprocess_fn) if preprocess_fn else dict(rescale=1. / 255)

    train_datagen = ImageDataGenerator(
        **common_kwargs,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30,
        brightness_range=(0.8, 1.2),
        zoom_range=0.1,
        validation_split=val_from_train,
    )
    test_datagen = ImageDataGenerator(**common_kwargs)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col="filepath", y_col="label", classes=classes,
        target_size=(img_size, img_size), batch_size=batch_size,
        class_mode="categorical", subset="training", seed=seed)

    val_gen = train_datagen.flow_from_dataframe(
        train_df, x_col="filepath", y_col="label", classes=classes,
        target_size=(img_size, img_size), batch_size=batch_size,
        class_mode="categorical", subset="validation", seed=seed)

    test_gen = test_datagen.flow_from_dataframe(
        test_df, x_col="filepath", y_col="label", classes=classes,
        target_size=(img_size, img_size), batch_size=batch_size,
        class_mode="categorical", shuffle=False)

    return train_gen, val_gen, test_gen


def compute_class_weights(df, classes):
    """Inverse-frequency class weights, to handle imbalance as the paper does."""
    from sklearn.utils.class_weight import compute_class_weight
    y = df["label"].values
    weights = compute_class_weight(class_weight="balanced", classes=np.array(classes), y=y)
    return {i: w for i, w in enumerate(weights)}
