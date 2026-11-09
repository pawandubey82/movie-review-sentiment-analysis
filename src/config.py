"""Central settings: file paths, constants and hyperparameter grids."""
from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "IMDB Dataset.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODEL_PATH = OUTPUT_DIR / "best_sentiment_model.pkl"      # best model (by CV F1)
ALL_MODELS_PATH = OUTPUT_DIR / "all_models.pkl"           # all tuned models
CV_RESULTS_PATH = OUTPUT_DIR / "cv_results.csv"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

# ---------- Data ----------
RANDOM_STATE = 42
TEST_SIZE = 0.20
LABEL_MAP = {"negative": 0, "positive": 1}
LABEL_NAMES = ["Negative", "Positive"]

# ---------- TF-IDF ----------
TFIDF_PARAMS = {
    "stop_words": "english",
    "max_features": 50000,
    "sublinear_tf": True,
}

# ---------- Models + GridSearch grids ----------
CV_FOLDS = 3
CV_SCORING = "f1"
