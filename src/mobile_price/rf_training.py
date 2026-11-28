import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    from .data_preprocessing import clean_data
    from .utils import get_project_root, load_object, read_yaml, save_object
except ImportError:
    from data_preprocessing import clean_data
    from utils import get_project_root, load_object, read_yaml, save_object


SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
TARGET_NAMES = ["0 (Low)", "1 (Medium)", "2 (High)", "3 (Very High)"]


def _cross_validation_scores(
    estimator: Any,
    features: pd.DataFrame,
    target: pd.Series,
    cv: StratifiedKFold,
) -> dict[str, float]:
    scores = cross_validate(
        estimator,
        features,
        target,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
        return_train_score=False,
    )
    return {
        "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
        "cv_accuracy_std": float(scores["test_accuracy"].std()),
        "cv_f1_macro_mean": float(scores["test_f1_macro"].mean()),
        "cv_f1_macro_std": float(scores["test_f1_macro"].std()),
    }


def run_random_forest_experiments(
    data_path: str | Path | None = None,
) -> dict[str, Any]:
    root = get_project_root()
    config = read_yaml(root / "config" / "params.yaml")
    data_config = config.get("data", {})
    rf_config = config.get("random_forest", {})

    raw_path = root / data_config.get("raw_path", "data/raw/train.csv")
    processed_path = Path(data_path) if data_path else root / data_config.get(
        "processed_path", "data/processed/train_cleaned.csv"
    )
    if not processed_path.exists():
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {raw_path}. Download Kaggle's train.csv "
                "and place it at data/raw/train.csv."
            )
        clean_data(raw_path, processed_path)

    df = pd.read_csv(processed_path)
    X = df.drop(columns=["price_range"])
    y = df["price_range"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(data_config.get("test_size", 0.2)),
        random_state=int(data_config.get("random_state", 42)),
        stratify=y,
    )

    scaler_path = root / config.get("model", {}).get(
        "scaler_path", "models/scaler.joblib"
    )
    scaler = load_object(scaler_path)
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    random_state = int(rf_config.get("random_state", 42))
    cv = StratifiedKFold(
        n_splits=int(rf_config.get("cv_splits", 5)),
        shuffle=True,
        random_state=random_state,
    )
    progress: list[dict[str, Any]] = []

    def record_progress(stage: str, estimator: Any, **parameters: Any) -> dict[str, Any]:
        row = {
            "stage": stage,
            **parameters,
            **_cross_validation_scores(estimator, X_train_scaled, y_train, cv),
        }
        progress.append(row)
        return row

    tree_depths = rf_config.get("tree_depths", [1, 2, 3, 4, 5, 8, None])
    tree_rows = []
    for depth in tree_depths:
        estimator = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=1, random_state=random_state
        )
        tree_rows.append(
            record_progress(
                f"Decision Tree depth={depth}",
                estimator,
                algorithm="Decision Tree",
                max_depth=depth,
                min_samples_leaf=1,
            )
        )

    best_tree_depth = max(tree_rows, key=lambda row: row["cv_accuracy_mean"])[
        "max_depth"
    ]
    for leaf_size in rf_config.get("tree_min_samples_leaf", [2, 5, 10]):
        estimator = DecisionTreeClassifier(
            max_depth=best_tree_depth,
            min_samples_leaf=leaf_size,
            random_state=random_state,
        )
        record_progress(
            f"Decision Tree depth={best_tree_depth}, leaf={leaf_size}",
            estimator,
            algorithm="Decision Tree",
            max_depth=best_tree_depth,
            min_samples_leaf=leaf_size,
        )

    for n_estimators in rf_config.get(
        "forest_estimators_progression", [1, 5, 10, 25, 50, 100]
    ):
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=1,
        )
        record_progress(
            f"Random Forest trees={n_estimators}",
            estimator,
            algorithm="Random Forest",
            n_estimators=n_estimators,
            max_depth=None,
            max_features="sqrt",
            min_samples_leaf=1,
        )

    search_config = rf_config.get("grid_search", {})
    param_grid = {
        "max_depth": search_config.get("max_depth", [None, 8, 16]),
        "max_features": search_config.get("max_features", ["sqrt", 0.7]),
        "min_samples_leaf": search_config.get("min_samples_leaf", [1, 2]),
    }
    search = GridSearchCV(
        RandomForestClassifier(
            n_estimators=int(search_config.get("n_estimators", 100)),
            random_state=random_state,
            n_jobs=1,
        ),
        param_grid=param_grid,
        scoring=SCORING,
        refit="accuracy",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(X_train_scaled, y_train)
    best_model = search.best_estimator_

    best_cv_row = search.cv_results_
    best_index = search.best_index_
    progress.append(
        {
            "stage": "Random Forest GridSearchCV best",
            "algorithm": "Random Forest",
            **search.best_params_,
            "n_estimators": int(search_config.get("n_estimators", 100)),
            "cv_accuracy_mean": float(best_cv_row["mean_test_accuracy"][best_index]),
            "cv_accuracy_std": float(best_cv_row["std_test_accuracy"][best_index]),
            "cv_f1_macro_mean": float(best_cv_row["mean_test_f1_macro"][best_index]),
            "cv_f1_macro_std": float(best_cv_row["std_test_f1_macro"][best_index]),
        }
    )

    y_pred = best_model.predict(X_test_scaled)
    svm_model_path = root / config.get("model", {}).get(
        "model_path", "models/best_svm_model.joblib"
    )
    svm_model = load_object(svm_model_path)
    svm_predictions = svm_model.predict(X_test_scaled.to_numpy())
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0,
    )
    final_metrics = {
        "model": "RandomForestClassifier",
        "best_params": search.best_params_,
        "cv": {
            "accuracy_mean": float(
                search.cv_results_["mean_test_accuracy"][best_index]
            ),
            "accuracy_std": float(
                search.cv_results_["std_test_accuracy"][best_index]
            ),
            "f1_macro_mean": float(
                search.cv_results_["mean_test_f1_macro"][best_index]
            ),
            "f1_macro_std": float(
                search.cv_results_["std_test_f1_macro"][best_index]
            ),
        },
        "test": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(
                y_test, y_pred, labels=[0, 1, 2, 3]
            ).tolist(),
        },
        "comparison_with_linear_svm": {
            "svm_C": float(svm_model.C),
            "svm_test_accuracy": float(accuracy_score(y_test, svm_predictions)),
            "svm_test_f1_macro": float(
                f1_score(y_test, svm_predictions, average="macro")
            ),
            "rf_accuracy_difference": float(
                accuracy_score(y_test, y_pred)
                - accuracy_score(y_test, svm_predictions)
            ),
            "rf_f1_macro_difference": float(
                f1_score(y_test, y_pred, average="macro")
                - f1_score(y_test, svm_predictions, average="macro")
            ),
        },
    }

    model_path = root / rf_config.get("model_path", "models/rf_model.joblib")
    metrics_path = root / rf_config.get(
        "metrics_path", "reports/metrics_rf.json"
    )
    progress_path = root / rf_config.get(
        "progress_path", "reports/rf_progress.csv"
    )
    cv_results_path = root / rf_config.get(
        "cv_results_path", "reports/rf_cv_results.csv"
    )
    comparison_path = root / rf_config.get(
        "comparison_path", "reports/rf_svm_feature_comparison.csv"
    )
    figure_dir = root / rf_config.get("figure_dir", "reports/figures")
    save_object(best_model, model_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(final_metrics, metrics_file, indent=2, ensure_ascii=False)
    progress_df = pd.DataFrame(progress)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_df.to_csv(progress_path, index=False)
    cv_results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search.cv_results_).sort_values("rank_test_accuracy").to_csv(
        cv_results_path, index=False
    )
    comparison = _compare_feature_importance(
        best_model,
        svm_model,
        X.columns,
    )
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)

    figure_dir.mkdir(parents=True, exist_ok=True)
    _save_progress_plot(progress_df, figure_dir / "rf_training_progress.png")
    _save_confusion_matrix(
        y_test,
        y_pred,
        figure_dir / "rf_confusion_matrix.png",
    )
    _save_feature_importance(
        best_model,
        X.columns,
        figure_dir / "rf_feature_importance.png",
    )
    _save_feature_comparison(
        comparison,
        figure_dir / "rf_svm_feature_comparison.png",
    )

    print(
        "[RF] Test accuracy: "
        f"{final_metrics['test']['accuracy']:.4f}; "
        f"macro F1: {final_metrics['test']['f1_macro']:.4f}"
    )
    return {
        "metrics": final_metrics,
        "progress": progress_df,
        "cv_results": pd.DataFrame(search.cv_results_),
        "feature_comparison": comparison,
        "model_path": model_path,
    }


def _save_progress_plot(progress: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 6))
    positions = range(len(progress))
    axis.plot(
        positions,
        progress["cv_accuracy_mean"],
        marker="o",
        label="CV accuracy",
    )
    axis.plot(
        positions,
        progress["cv_f1_macro_mean"],
        marker="s",
        label="CV macro F1",
    )
    axis.set_xticks(list(positions))
    axis.set_xticklabels(progress["stage"], rotation=55, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean score (Stratified 5-Fold CV)")
    axis.set_title("Decision Tree and Random Forest experiment progression")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _save_confusion_matrix(
    y_true: pd.Series,
    y_pred: Any,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]),
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
        ax=axis,
    )
    axis.set_title("Random Forest confusion matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _save_feature_importance(
    model: RandomForestClassifier,
    feature_names: pd.Index,
    output_path: Path,
) -> None:
    importance = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values()
    top_features = importance.tail(10)

    figure, axis = plt.subplots(figsize=(8, 6))
    top_features.plot.barh(ax=axis, color="forestgreen")
    axis.set_title("Top 10 Random Forest impurity-based feature importances")
    axis.set_xlabel("Mean decrease in impurity")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _compare_feature_importance(
    forest: RandomForestClassifier,
    svm_model: Any,
    feature_names: pd.Index,
) -> pd.DataFrame:
    rf_importance = pd.Series(
        forest.feature_importances_, index=feature_names
    )
    svm_importance = pd.Series(
        np.mean(np.abs(svm_model.coef_), axis=0).ravel(),
        index=feature_names,
    )
    comparison = pd.DataFrame(
        {
            "feature": feature_names,
            "rf_impurity_importance": rf_importance.values,
            "svm_mean_abs_pairwise_coefficient": svm_importance.values,
            "rf_rank": rf_importance.rank(ascending=False, method="min").values,
            "svm_rank": svm_importance.rank(
                ascending=False, method="min"
            ).values,
        }
    )
    comparison["average_rank"] = (
        comparison["rf_rank"] + comparison["svm_rank"]
    ) / 2
    return comparison.sort_values("average_rank").reset_index(drop=True)


def _save_feature_comparison(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    top_features = comparison.head(10).sort_values("average_rank", ascending=True)
    positions = np.arange(len(top_features))
    bar_height = 0.36

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        positions - bar_height / 2,
        top_features["rf_rank"],
        bar_height,
        label="Random Forest rank",
        color="forestgreen",
    )
    axis.barh(
        positions + bar_height / 2,
        top_features["svm_rank"],
        bar_height,
        label="Linear SVM rank",
        color="steelblue",
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(top_features["feature"])
    axis.invert_yaxis()
    axis.set_xlabel("Rank (1 = most important)")
    axis.set_title("Top features: Random Forest vs Linear SVM")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    run_random_forest_experiments()
