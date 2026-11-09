"""Predict sentiment for new reviews using the saved best model.

Usage:
    python -m src.predict "This movie was fantastic!"
    python -m src.predict            # interactive mode
"""
import sys

import joblib

from src import config
from src.data_preprocessing import clean_review


def load_model(path=config.MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python main.py --step train` first.")
    return joblib.load(path)


def predict_sentiment(texts, model=None):
    """Return a list of dicts: {review, sentiment, confidence/score}."""
    model = model or load_model()
    if isinstance(texts, str):
        texts = [texts]
    cleaned = [clean_review(t) for t in texts]
    labels = model.predict(cleaned)

    results = []
    if hasattr(model, "predict_proba"):
        extra = ("confidence", model.predict_proba(cleaned).max(axis=1))
    else:  # e.g. LinearSVC: signed distance from the decision boundary
        extra = ("score", model.decision_function(cleaned))

    for text, label, value in zip(texts, labels, extra[1]):
        results.append({
            "review": text,
            "sentiment": config.LABEL_NAMES[int(label)],
            extra[0]: round(float(abs(value) if extra[0] == "score" else value), 3),
        })
    return results


def main():
    model = load_model()
    if len(sys.argv) > 1:
        for r in predict_sentiment(sys.argv[1:], model):
            print(r)
        return
    print("Type a movie review (empty line to quit).")
    while True:
        text = input("> ").strip()
        if not text:
            break
        print(predict_sentiment(text, model)[0])


if __name__ == "__main__":
    main()
