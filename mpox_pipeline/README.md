# Monkeypox Detection Pipeline (paper-faithful replication)

## Setup
```bash
pip install tensorflow scikit-learn pandas numpy scipy matplotlib opencv-python shap lime albumentations
```

## Folder structure expected
```
your_dataset/
    Monkeypox/*.jpg
    Chickenpox/*.jpg
    Measles/*.jpg
    ...
```
Rename `DATA_DIR` in `train.py` to point to this folder.

## Run order

1. **Sanity check your data**
   ```python
   from data_utils import build_dataframe
   df, classes = build_dataframe("path/to/your_dataset")
   ```
   Confirm class counts look right and check for imbalance (paper had monkeypox at 37.6%,
   others as low as 7.3% — decide whether you need `compute_class_weights` or augmentation
   like the paper, or both).

2. **Ablation study (`train.py`)**
   Trains every (backbone x attention x depth) config across 5 folds.
   - **Full replication is expensive**: 10 backbones x 4 attention types x up to 2 depths
     x 5 folds. Trim `BACKBONE_LIST` / `ATTENTION_TYPES` / `DEPTHS` at the top of the file
     if your GPU/time budget is limited — e.g. start with 3-4 backbones and CBAM only,
     confirm the pipeline works end-to-end, then scale up.
   - Logs results to `results/results.json` after every config (crash-safe).

3. **Pick top-5 per attention type**
   From `results/results.json`, sort by `mean_accuracy` within each attention type
   and take the top 5 — this mirrors the paper's Table 3-5 -> Table 6-8 step.

4. **Ensemble search (`ensemble.py`)**
   For each attention type's top-5, get each model's predicted probabilities on a held-out
   test set (or per fold), then call `search_best_triple()` to try all 10 combinations of 3
   with both `method="average"` and `method="majority"`. This reproduces Tables 6-8 and
   picks your candidate "proposed model."

5. **Statistical validation (`evaluate.py`)**
   Use `paired_significance_test()` on per-fold accuracy/F1 arrays: ensemble vs. each
   individual base model, to confirm the improvement is statistically significant
   (paper's Table 12, paired t-test + 95% CI).

6. **Explainability (`xai.py`)**
   `grad_cam()` is fully implemented and ready to use on any trained model. For SHAP and
   LIME, see the commented usage blocks at the bottom of the file — both need `pip install
   shap lime` and work directly on your trained Keras models.

## Notes / deviations you should decide on
- The paper freezes backbones for "base model" evaluation, then presumably still keeps
  them frozen when attention is added (transfer learning, not fine-tuning) — `fine_tune=False`
  in `build_model()` matches this. If your dataset is small (a few hundred images, like
  yours), keep it frozen — fine-tuning risks overfitting on a small set.
- The paper's dense head (1024→256→128) is heavy relative to a small dataset; if you
  overfit, reduce it (e.g. 256→64) — note this as a deviation in your methodology.
- Batch size 32, lr 1e-4, Adam, categorical cross-entropy, 50 epochs w/ early stopping —
  all matched to the paper's settings in `train.py`.
