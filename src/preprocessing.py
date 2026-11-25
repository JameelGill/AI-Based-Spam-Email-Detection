
import pandas as pd
import re
import html
from pathlib import Path

# Project folder location
BASE_DIR = Path(__file__).resolve().parent.parent

# Input and output files
INPUT_FILE = BASE_DIR / "dataset" / "processed_emails.csv"
OUTPUT_FILE = BASE_DIR / "dataset" / "cleaned_emails.csv"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Keep letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Original emails:", len(df))

    # Combine subject and email body
    df["text"] = (
        df["subject"].fillna("").astype(str)
        + " "
        + df["email_text"].fillna("").astype(str)
    )

    print("Cleaning email text...")

    df["cleaned_text"] = df["text"].apply(clean_text)

    # Remove empty emails
    df = df[df["cleaned_text"].str.strip() != ""]

    # Keep only required columns
    df = df[["cleaned_text", "label"]]

    # Save cleaned dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nPreprocessing completed!")
    print("Total cleaned emails:", len(df))
    print("Ham emails:", (df["label"] == 0).sum())
    print("Spam emails:", (df["label"] == 1).sum())
    print("Saved file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
