def compute_best_threshold(preds, targs, bins=100):
    """
    Computes the best threshold for a binary classification task
    across multiple metrics. Note that average_precision and roc_auc scores are threshold-invariant.

    Args:
        preds (np.ndarray): Predicted scores/probabilities (shape: [N]).
        targs (np.ndarray): Ground truth binary labels (0/1) (shape: [N]).
        bins (int): Number of threshold steps to evaluate between 0 and 1.

    Returns:
        dict: Best threshold per metric.
    """
    import numpy as np
    import polars as pl
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    thresholds = np.linspace(0.0, 1.0, bins + 1)

    best = {
        "accuracy": (0, -np.inf),
        "precision": (0, -np.inf),
        "recall": (0, -np.inf),
        "f1": (0, -np.inf),
        "mcc": (0, -np.inf),
        "balanced_accuracy": (0, -np.inf),
    }

    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        elif hasattr(x, "numpy"):
            return x.numpy()
        else:
            return np.asarray(x)

    preds = to_numpy(preds)
    targs = to_numpy(targs)

    for t in thresholds:
        preds_bin = (preds >= t).astype(int)

        for metric in best:
            try:
                if metric == "accuracy":
                    score = accuracy_score(targs, preds_bin)
                elif metric == "precision":
                    score = precision_score(targs, preds_bin, zero_division=0)
                elif metric == "recall":
                    score = recall_score(targs, preds_bin, zero_division=0)
                elif metric == "f1":
                    score = f1_score(targs, preds_bin, zero_division=0)
                elif metric == "mcc":
                    score = matthews_corrcoef(targs, preds_bin)
                elif metric == "balanced_accuracy":
                    score = balanced_accuracy_score(targs, preds_bin)

                else:
                    continue  # skip unknown metric

                if score > best[metric][1]:
                    best[metric] = (t, score)
            except Exception:
                continue  # skip this threshold for this metric if error

    df = pl.DataFrame(
        data=[
            (metric, score, threshold) for metric, (threshold, score) in best.items()
        ],
        schema=["metric", "score", "best_threshold"],
    )
    return df


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    X, y = make_classification(n_samples=1000, weights=[0.7, 0.3])
    model = LogisticRegression().fit(X, y)
    probs = model.predict_proba(X)[:, 1]

    thresholds = compute_best_threshold(probs, y, bins=10)
    print(thresholds)
