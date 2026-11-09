"""Test-set evaluation: metrics, confusion matrices, ROC curves, feature importance."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src import config


def _save(name: str):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / name, dpi=150)
    plt.close()


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate_models(models: dict, X_test, y_test):
    """Returns (results_df, predictions, scores)."""
    rows, preds, scores = [], {}, {}
    for name, model in models.items():
        pred = model.predict(X_test)
        score = _scores(model, X_test)
        preds[name], scores[name] = pred, score
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, score),
        })
    results = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    return results, preds, scores


def plot_confusion_matrices(preds: dict, y_test):
    for name, pred in preds.items():
        print(f"\n{'=' * 50}\nModel: {name}")
        print(classification_report(y_test, pred, target_names=config.LABEL_NAMES, zero_division=0))
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=config.LABEL_NAMES, yticklabels=config.LABEL_NAMES)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted Sentiment")
        plt.ylabel("Actual Sentiment")
        _save(f"confusion_matrix_{name.lower().replace(' ', '_')}.png")


def plot_model_comparison(results: pd.DataFrame):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    results.set_index("Model")[metrics].plot(kind="bar", figsize=(12, 6), ylim=(0, 1), rot=0)
    plt.title("Performance Comparison of ML Models")
    plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    _save("model_comparison.png")


def plot_roc_curves(scores: dict, y_test):
    plt.figure(figsize=(8, 6))
    for name, score in scores.items():
        fpr, tpr, _ = roc_curve(y_test, score)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Sentiment Classification")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    _save("roc_curves.png")


def plot_feature_importance(models: dict, n=15):
    """Top positive/negative words from the Logistic Regression coefficients."""
    if "Logistic Regression" not in models:
        return
    pipe = models["Logistic Regression"]
    df = pd.DataFrame({
        "Word": pipe.named_steps["tfidf"].get_feature_names_out(),
        "Coefficient": pipe.named_steps["classifier"].coef_[0],
    })
    top_pos, top_neg = df.nlargest(n, "Coefficient"), df.nsmallest(n, "Coefficient")
    print("\nTop positive features:\n", top_pos.to_string(index=False))
    print("\nTop negative features:\n", top_neg.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(top_pos["Word"][::-1], top_pos["Coefficient"][::-1])
    axes[0].set_title("Top Positive Sentiment Features")
    axes[1].barh(top_neg["Word"][::-1], top_neg["Coefficient"][::-1])
    axes[1].set_title("Top Negative Sentiment Features")
    for ax in axes:
        ax.set_xlabel("Coefficient")
    _save("feature_importance.png")


def run_evaluation(models: dict, X_test, y_test):
    """Evaluate all models, save tables + figures."""
    print("\n=== TEST SET EVALUATION ===")
    results, preds, scores = evaluate_models(models, X_test, y_test)
    print(results.round(4).to_string(index=False))

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.COMPARISON_PATH, index=False)

    pred_df = pd.DataFrame({"review": X_test.values, "actual": y_test.values})
    for name, pred in preds.items():
        pred_df[f"pred_{name}"] = pred
    pred_df.to_csv(config.PREDICTIONS_PATH, index=False)

    plot_confusion_matrices(preds, y_test)
    plot_model_comparison(results)
    plot_roc_curves(scores, y_test)
    plot_feature_importance(models)
    print(f"\nSaved: {config.COMPARISON_PATH.name}, {config.PREDICTIONS_PATH.name}, figures in {config.FIGURES_DIR}")
    return results
