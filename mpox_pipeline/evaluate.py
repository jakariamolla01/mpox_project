"""
7 metrics used in the paper: Accuracy, Precision, Recall, F1, Specificity, MCC, Youden's J.
Plus paired statistical testing (paired t-test across folds) for ensemble vs base models.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, matthews_corrcoef, confusion_matrix)
from scipy import stats


def compute_metrics(y_true, y_pred, average="weighted"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    # macro-averaged specificity across classes (one-vs-rest)
    specs = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    specificity = float(np.mean(specs))
    youden_j = rec + specificity - 1

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "specificity": specificity, "mcc": mcc, "youden_j": youden_j,
    }


def paired_significance_test(scores_a, scores_b, alpha=0.05):
    """
    scores_a / scores_b: per-fold metric arrays (e.g. accuracy across 5 folds)
    for the ensemble vs. one base model. Paired t-test, matching paper's Table 12.
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    ci = stats.t.interval(1 - alpha, len(scores_a) - 1,
                           loc=np.mean(scores_a), scale=stats.sem(scores_a))
    return {
        "mean_a": float(np.mean(scores_a)), "std_a": float(np.std(scores_a)),
        "ci_95": ci, "p_value": float(p_value),
        "significant": p_value < alpha,
    }
