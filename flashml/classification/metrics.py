def compute_binary_classification_metrics(
    scores, target, threshold=0.5, average="binary"
):
    """
    Compute metrics for binary classification from score probabilities.

    Args:
        scores (array-like): Scores/probabilities for the positive class.
                             Shape: (n_samples,) or (n_samples, 2) (then second column used).
        target (array-like): True binary labels (0 or 1).
        threshold (float): Threshold to convert scores to predicted labels.
        average (str): Averaging method for precision, recall, f1 (default "binary").

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    scores = np.asarray(scores)
    target = np.asarray(target)

    if scores.ndim == 2 and scores.shape[1] == 2:
        pos_scores = scores[:, 1]
    else:
        pos_scores = scores.ravel()

    predicted = (pos_scores >= threshold).astype(int)

    metrics = {}
    metrics["accuracy"] = accuracy_score(target, predicted)
    metrics["precision"] = precision_score(
        target, predicted, average=average, zero_division=0
    )
    metrics["recall"] = recall_score(
        target, predicted, average=average, zero_division=0
    )
    metrics["f1"] = f1_score(target, predicted, average=average, zero_division=0)

    try:
        metrics["roc_auc"] = roc_auc_score(target, pos_scores)
        metrics["pr_auc"] = average_precision_score(target, pos_scores)
    except Exception:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    try:
        metrics["mcc"] = matthews_corrcoef(target, predicted)
    except Exception:
        metrics["mcc"] = float("nan")

    metrics["balanced_accuracy"] = balanced_accuracy_score(target, predicted)
    return metrics


def compute_multiclass_classification_metrics(scores, target, average="macro"):
    """
    Compute metrics for multiclass classification from score probabilities.

    Args:
        scores (array-like): Score probabilities. Shape: (n_samples, n_classes).
        target (array-like): True labels.
        average (str): Averaging method for precision, recall, f1 (default "macro").

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize

    scores = np.asarray(scores)
    target = np.asarray(target)
    classes = np.unique(target)

    predicted = np.argmax(scores, axis=1)

    metrics = {}
    metrics["accuracy"] = accuracy_score(target, predicted)
    metrics["precision"] = precision_score(
        target, predicted, average=average, zero_division=0
    )
    metrics["recall"] = recall_score(
        target, predicted, average=average, zero_division=0
    )
    metrics["f1"] = f1_score(target, predicted, average=average, zero_division=0)

    try:
        target_bin = label_binarize(target, classes=classes)
        metrics["roc_auc"] = roc_auc_score(
            target_bin, scores, average=average, multi_class="ovr"
        )
        metrics["pr_auc"] = average_precision_score(target_bin, scores, average=average)
    except Exception:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    try:
        metrics["mcc"] = matthews_corrcoef(target, predicted)
    except Exception:
        metrics["mcc"] = float("nan")

    metrics["balanced_accuracy"] = balanced_accuracy_score(target, predicted)
    return metrics
