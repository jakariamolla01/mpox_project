"""
Orchestrates the ablation study: trains every (backbone, attention_type, attention_depth)
config across 5 folds, logs per-fold metrics to results.csv.
"""
import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Ensure local imports work correctly regardless of execution path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_builder import build_model, BACKBONES
from data_utils import build_dataframe, get_kfold_splits, make_generators, compute_class_weights
from evaluate import compute_metrics

# --- GPU setup for 8GB cards (RTX 2070 Super etc.) ---
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)  # don't grab all VRAM at once
    if gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")  # ~2x memory headroom
except Exception as e:
    print(f"[Notice] GPU config warning: {e}")

SEED = 108
BACKBONE_LIST = list(BACKBONES.keys())      # trim if needed, e.g. ["ResNet50V2", "DenseNet121"]
ATTENTION_TYPES = ["none", "eca", "se", "cbam"]
DEPTHS = ["single", "double"]

tf.random.set_seed(SEED)
np.random.seed(SEED)


def train_one_config(backbone_name, attention_type, attention_depth, df, classes, splits, output_dir, batch_size, epochs):
    fold_metrics = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        model, preprocess_fn, size = build_model(
            backbone_name, num_classes=len(classes),
            attention_type=attention_type, attention_depth=attention_depth)

        train_gen, val_gen, test_gen = make_generators(
            df, train_idx, test_idx, classes, size, batch_size, preprocess_fn)

        class_weights = compute_class_weights(df.iloc[train_idx], classes)

        model.compile(optimizer=Adam(learning_rate=1e-4),
                      loss="categorical_crossentropy", metrics=["accuracy"])

        ckpt_path = os.path.join(output_dir, f"{backbone_name}_{attention_type}_{attention_depth}_fold{fold_i}.keras")
        callbacks = [
            EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-7),
            ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
        ]

        model.fit(train_gen, validation_data=val_gen, epochs=epochs,
                  class_weight=class_weights, callbacks=callbacks, verbose=1)

        y_prob = model.predict(test_gen)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = test_gen.classes
        m = compute_metrics(y_true, y_pred)
        m["fold"] = fold_i
        fold_metrics.append(m)
        print(f"[{backbone_name}-{attention_type}-{attention_depth}] fold {fold_i}: acc={m['accuracy']:.4f}")

        tf.keras.backend.clear_session()

    return fold_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Mpox Pipeline")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df, classes = build_dataframe(args.data_dir)
    splits = get_kfold_splits(df, n_splits=5, seed=SEED)

    all_results = []
    for backbone_name in BACKBONE_LIST:
        for attention_type in ATTENTION_TYPES:
            depths = ["single"] if attention_type == "none" else DEPTHS
            for depth in depths:
                fold_metrics = train_one_config(
                    backbone_name, attention_type, depth, df, classes, splits,
                    args.output_dir, args.batch_size, args.epochs
                )
                mean_acc = np.mean([m["accuracy"] for m in fold_metrics])
                all_results.append({
                    "backbone": backbone_name, "attention": attention_type, "depth": depth,
                    "mean_accuracy": mean_acc, "fold_metrics": fold_metrics,
                })
                with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                    json.dump(all_results, f, indent=2)

    print("Done. See results/results.json")


if __name__ == "__main__":
    main()