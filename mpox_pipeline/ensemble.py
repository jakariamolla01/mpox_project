"""
Builds average and majority-voting ensembles from a set of trained models,
and searches all C(5,3) combinations among the top-5 models per attention type
(mirroring Tables 6-8 of the paper).
"""
import itertools
import numpy as np
from scipy import stats
from evaluate import compute_metrics


def average_ensemble_predict(models_probs):
    """models_probs: list of (N, num_classes) prob arrays from different models on the same data."""
    return np.mean(np.stack(models_probs, axis=0), axis=0)


def majority_vote_predict(models_probs):
    """
    models_probs: list of (N, num_classes) prob arrays.
    Each model casts a vote (argmax); ties broken by highest average probability
    among tied classes, matching Eq. 12 and the paper's tie-break rule.
    """
    preds = np.stack([np.argmax(p, axis=1) for p in models_probs], axis=0)  # (M, N)
    avg_probs = average_ensemble_predict(models_probs)                      # (N, C)
    n_samples = preds.shape[1]
    final = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        votes = preds[:, i]
        counts = np.bincount(votes, minlength=avg_probs.shape[1])
        max_votes = counts.max()
        tied_classes = np.where(counts == max_votes)[0]
        if len(tied_classes) == 1:
            final[i] = tied_classes[0]
        else:
            final[i] = tied_classes[np.argmax(avg_probs[i, tied_classes])]
    return final


def evaluate_ensemble(model_probs_dict, y_true, method="majority"):
    """model_probs_dict: {model_name: probs_array}"""
    probs_list = list(model_probs_dict.values())
    if method == "average":
        avg = average_ensemble_predict(probs_list)
        y_pred = np.argmax(avg, axis=1)
    else:
        y_pred = majority_vote_predict(probs_list)
    return compute_metrics(y_true, y_pred)


def search_best_triple(top5_probs_dict, y_true, method="majority"):
    """
    top5_probs_dict: {model_name: probs_array} for the top-5 models of one attention type.
    Tries all C(5,3)=10 combinations, returns sorted results (best first).
    """
    names = list(top5_probs_dict.keys())
    results = []
    for combo in itertools.combinations(names, 3):
        subset = {n: top5_probs_dict[n] for n in combo}
        metrics = evaluate_ensemble(subset, y_true, method=method)
        results.append({"combo": combo, **metrics})
    results.sort(key=lambda r: r["accuracy"], reverse=True)
    return results
