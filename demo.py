"""Live demo: type a short English article text, get its Wikipedia article type.

Run from the project folder:   python demo.py
Type 'q' (or press Enter on an empty line) to quit.
"""
import os
import sys
import warnings

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
from preprocessing import preprocess  # noqa: E402  (same cleaning as training)

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "bow_classifier.joblib")
TRAIN_CSV = os.path.join(PROJECT_DIR, "data", "train.csv")


def load_model():
    """Load the saved model; if it was saved with another scikit-learn version,
    retrain the same pipeline from the frozen training data (about 1 minute)."""
    import joblib
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # treat version-mismatch warnings as failures
            return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Saved model could not be loaded ({type(e).__name__}); retraining from data/train.csv ...")
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    train = pd.read_csv(TRAIN_CSV)
    model = Pipeline([("bow", CountVectorizer(min_df=2)),
                      ("clf", LogisticRegression(max_iter=3000, C=1.0))])
    model.fit(train.text.apply(lambda t: preprocess(t, as_string=True)), train.label_name)
    return model


def predict_article_type(model, text):
    probs = model.predict_proba([preprocess(text, as_string=True)])[0]
    return sorted(zip(model.classes_, probs), key=lambda x: -x[1])


def main():
    print("Loading model ...")
    model = load_model()
    print("\nWikipedia article-type classifier")
    print("Classes:", ", ".join(model.classes_))
    print("Type an English article text (or 'q' to quit).\n")
    while True:
        try:
            text = input("Text> ").strip()
        except EOFError:
            break
        if not text or text.lower() in ("q", "quit", "exit"):
            break
        ranked = predict_article_type(model, text)
        print(f"  Prediction: {ranked[0][0]}  ({ranked[0][1]:.1%})")
        print("  Top 3: " + ", ".join(f"{c} {p:.1%}" for c, p in ranked[:3]) + "\n")
    print("Bye!")


if __name__ == "__main__":
    main()
