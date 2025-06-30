from typing import Literal


def run_dummy_regressors(
    X_train,
    Y_train,
    X_test,
    Y_test,
    run_name: str = "baselines-regression",
    experiment_name: str = None,
):
    """
    Compute baseline metrics using all dummy strategies for regressor.
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
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        root_mean_squared_error,
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

    strategies = [
        ("mean", DummyRegressor(strategy="mean")),
        ("median", DummyRegressor(strategy="median")),
        ("quantile", DummyRegressor(strategy="quantile", quantile=0.5)),
        (
            "constant",
            DummyRegressor(strategy="constant", constant=np.mean(Y_train)),
        ),
    ]
    for strategy_name, dummy in strategies:
        dummy.fit(X_train, Y_train)
        Y_pred = dummy.predict(X_test)
        results.append(
            {
                "strategy_name": strategy_name,
                "mse": mean_squared_error(Y_test, Y_pred),
                "mae": mean_absolute_error(Y_test, Y_pred),
                "rmse": root_mean_squared_error(Y_test, Y_pred),
                "r2": r2_score(Y_test, Y_pred),
            }
        )

    import os
    import sys

    import mlflow

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from flashml.main_tools.mlflow_logging import _TrainingLogger

    just_to_initialize_mlflow = _TrainingLogger(
        None, run_name=run_name, experiment_name=experiment_name
    )
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
        # also add roc-auc and other stupid metrics


def run_linear_regressor(
    X_train,
    Y_train,
    X_test,
    Y_test,
    regularization: Literal["l1", "l2"] = None,
    run_name: str = "baselines-regression",
    experiment_name: str = None,
):
    """
    Train and evaluate a linear model (regression) with optional regularization.
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
        Lasso,
        LinearRegression,
        Ridge,
    )
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        root_mean_squared_error,
    )

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    results = []
    if regularization == "l2":
        model = Ridge()
    elif regularization == "l1":
        model = Lasso()
    else:
        model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    results.append(
        {
            "model": f"{regularization or 'linear'}_regression",
            "mse": mean_squared_error(Y_test, Y_pred),
            "mae": mean_absolute_error(Y_test, Y_pred),
            "rmse": root_mean_squared_error(Y_test, Y_pred),
            "r2": r2_score(Y_test, Y_pred),
        }
    )

    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from flashml.main_tools.mlflow_logging import _TrainingLogger

    just_to_initialize_mlflow = _TrainingLogger(
        None, run_name=run_name, experiment_name=experiment_name
    )
    for res in results:
        run_name = f"{res.pop('model')}"
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_metrics(res, synchronous=False)
