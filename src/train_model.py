
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "cleaned_emails.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "spam_model.pkl"


def main():

    print("Loading cleaned dataset...")

    if not DATA_FILE.exists():
        print("Error: cleaned_emails.csv not found!")
        print("Run preprocessing.py first.")
        return

    df = pd.read_csv(DATA_FILE)

    X = df["cleaned_text"].fillna("")
    y = df["label"]

    print("Total emails:", len(df))
    print("Ham emails:", (y == 0).sum())
    print("Spam emails:", (y == 1).sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2)
        )),
        ("classifier", ComplementNB(alpha=0.5))
    ])

    print("\nTraining AI model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Training Completed!")
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Ham", "Spam"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    print("\nModel saved successfully!")
    print("Location:", MODEL_FILE)


if __name__ == "__main__":
    main()
