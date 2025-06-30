from typing import Literal


def run_dummy_classifiers(
    X_train,
    Y_train,
    X_test,
    Y_test,
    average="macro",
    run_name: str = "baselines-classification",
    experiment_name: str = None,
):
    """
    Compute baseline metrics using all dummy strategies for classification.
    Args:
        X_train (ndarray, torch.tensor): Training features
        Y_train (ndarray, torch.tensor): Training targets
        X_test (ndarray, torch.tensor): Test features
        Y_test (ndarray, torch.tensor): Test targets
        experiment_name: Optional name for the experiment
    Returns:
        list: List of dictionaries containing baseline metrics for each strategy
    """
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    # Convert to numpy if inputs are torch tensors
    # if isinstance(X_train, np.ndarray):
    #     X_train = X_train.cpu().numpy()
    # if isinstance(Y_train, np.ndarray):
    #     Y_train = Y_train.cpu().numpy()
    # if isinstance(X_test, np.ndarray):
    #     X_test = X_test.cpu().numpy()
    # if isinstance(Y_test, np.ndarray):
    #     Y_test = Y_test.cpu().numpy()
    #
    # Ensure inputs are numpy arrays
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)
    results = []
    most_frequent_class = np.bincount(Y_train.astype(int)).argmax()
    strategies = [
        ("most_frequent", DummyClassifier(strategy="most_frequent")),
        ("stratified", DummyClassifier(strategy="stratified")),
        ("uniform", DummyClassifier(strategy="uniform")),
        (
            "constant",
            DummyClassifier(strategy="constant", constant=most_frequent_class),
        ),
        (
            "prior",
            DummyClassifier(
                strategy="prior",
            ),
        ),
    ]
    # For binary and multiclass, roc_auc requires probability estimates for positive class
    for strategy_name, dummy in strategies:
        dummy.fit(X_train, Y_train)
        Y_pred = dummy.predict(X_test)
        try:
            Y_prob = dummy.predict_proba(X_test)
            roc_auc = roc_auc_score(Y_test, Y_prob, average=average, multi_class="ovo")
        except Exception:
            roc_auc = roc_auc_score(Y_test, Y_pred, average=average)

        results.append(
            {
                "strategy_name": strategy_name,
                "accuracy": accuracy_score(Y_test, Y_pred),
                "f1": f1_score(Y_test, Y_pred, average=average, zero_division=0),
                "recall": recall_score(
                    Y_test, Y_pred, average=average, zero_division=0
                ),
                "precision": precision_score(
                    Y_test, Y_pred, average=average, zero_division=0
                ),
                "roc_auc": roc_auc,
                "mcc": matthews_corrcoef(Y_test, Y_pred),
            }
        )
    import os
    import sys

    import mlflow

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from flashml.main_tools.mlflow_logging import _TrainingLogger

    _ = _TrainingLogger(
        None, run_name=run_name, experiment_name=experiment_name
    )  # just_to_initialize_mlflow
    for strat in results:
        with mlflow.start_run(
            run_name="dummy_"
            # + {"regression_" if is_regression else "classification_"}
            + strat["strategy_name"],
            nested=True,
        ):
            del strat["strategy_name"]
            mlflow.log_metrics(strat, step=None, synchronous=False)
        # add extras argument that also trains a linear classifier, maybe adaboost and other simple shit
        # or better just add log_linear_model_metrics, or just create a class named Baselines that allow all this shit together.


def run_linear_classifier(
    X_train,
    Y_train,
    X_test,
    Y_test,
    regularization: Literal["l1", "l2"] = None,
    average="macro",
    run_name: str = "baselines-classification",
    experiment_name: str = None,
):
    """
    Train and evaluate a linear model (regression or classification) with optional regularization.
    Args:
        X_train: Training features
        Y_train: Training targets
        X_test: Test features
        Y_test: Test targets
        regularization: Type of regularization ("l1" or "l2")
        average: Scoring average method for classification
        run_name: MLflow run name
        experiment_name: Optional MLflow experiment name
    Returns:
        list: List of dictionaries containing metrics for the linear model
    """
    import mlflow
    import numpy as np
    from sklearn.linear_model import (
        LogisticRegression,
        RidgeClassifier,
    )
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)
    results = []

    if regularization == "l2":
        model = RidgeClassifier()
    elif regularization == "l1":
        model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
    else:
        model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    try:
        Y_prob = model.predict_proba(X_test)
        roc_auc = roc_auc_score(Y_test, Y_prob, average=average, multi_class="ovo")
    except Exception:
        roc_auc = roc_auc_score(Y_test, Y_pred, average=average)
    results.append(
        {
            "model": f"{regularization or 'logistic'}_regression",
            "accuracy": accuracy_score(Y_test, Y_pred),
            "f1": f1_score(Y_test, Y_pred, average=average, zero_division=0),
            "recall": recall_score(Y_test, Y_pred, average=average, zero_division=0),
            "precision": precision_score(
                Y_test, Y_pred, average=average, zero_division=0
            ),
            "roc_auc": roc_auc,
            "mcc": matthews_corrcoef(Y_test, Y_pred),
        }
    )
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from flashml.main_tools.mlflow_logging import _TrainingLogger

    _ = _TrainingLogger(None, run_name=run_name, experiment_name=experiment_name)
    for res in results:
        run_name = f"{res.pop('model')}"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_metrics(res, synchronous=False)
