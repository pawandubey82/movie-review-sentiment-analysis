"""Baseline + model training with TF-IDF and GridSearchCV."""
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src import config


def get_model_definitions() -> dict:
    """Model name -> (classifier, GridSearch parameter grid)."""
    tfidf_grid = {"tfidf__min_df": [2, 5], "tfidf__ngram_range": [(1, 1), (1, 2)]}
    return {
        "Naive Bayes": (
            MultinomialNB(),
            {**tfidf_grid, "classifier__alpha": [0.5, 1.0]},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1500, random_state=config.RANDOM_STATE),
            {**tfidf_grid, "classifier__C": [0.5, 1.0, 2.0]},
        ),
        "Linear SVM": (
            LinearSVC(random_state=config.RANDOM_STATE),
            {**tfidf_grid, "classifier__C": [0.5, 1.0, 2.0]},
        ),
    }


def run_baseline(y_train, y_test):
    """Majority-class baseline, to show what 'no learning' scores."""
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    pred = baseline.predict(np.zeros((len(y_test), 1)))
    print("\n=== BASELINE (most frequent class) ===")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred, target_names=config.LABEL_NAMES, zero_division=0))


def train_all(X_train, y_train):
    """Tune every model with GridSearchCV. Returns (best_models, cv_df)."""
    best_models, rows = {}, []

    for name, (classifier, params) in get_model_definitions().items():
        print(f"\nTraining and tuning: {name}")
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**config.TFIDF_PARAMS)),
            ("classifier", classifier),
        ])
        search = GridSearchCV(pipeline, params, scoring=config.CV_SCORING,
                              cv=config.CV_FOLDS, n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)

        best_models[name] = search.best_estimator_
        rows.append({"Model": name, "Best CV F1": search.best_score_,
                     "Best Parameters": search.best_params_})
        print(f"Best CV F1: {search.best_score_:.4f}")
        print("Best parameters:", search.best_params_)

    cv_df = pd.DataFrame(rows).sort_values("Best CV F1", ascending=False).reset_index(drop=True)
    return best_models, cv_df


def save_models(best_models: dict, cv_df: pd.DataFrame):
    """Save all tuned models, the CV table, and the best model (chosen by CV F1)."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_models, config.ALL_MODELS_PATH)
    cv_df.to_csv(config.CV_RESULTS_PATH, index=False)

    best_name = cv_df.loc[0, "Model"]
    joblib.dump(best_models[best_name], config.MODEL_PATH)
    print(f"\nBest model by CV F1: {best_name}")
    print(f"Saved to: {config.MODEL_PATH}")
    return best_name


def load_all_models() -> dict:
    return joblib.load(config.ALL_MODELS_PATH)
