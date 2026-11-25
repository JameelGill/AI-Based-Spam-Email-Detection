
import os
import re
import pandas as pd
from pathlib import Path
from email import policy
from email.parser import BytesParser

# 1. Get project directory
project_dir = Path(__file__).resolve().parent.parent

# 2. Dataset folder paths
ham_dir = project_dir / "dataset" / "ham"
spam_dir = project_dir / "dataset" / "spam"


# 3. Function to extract email content
def extract_email(file_path):

    with open(file_path, "rb") as file:
        message = BytesParser(
            policy=policy.default
        ).parse(file)

    # Get email subject
    subject = str(message.get("subject", ""))

    # Get email body
    body = ""

    if message.is_multipart():

        for part in message.walk():

            if part.get_content_type() == "text/plain":

                if part.get_content_disposition() == "attachment":
                    continue

                try:
                    body += str(part.get_content()) + " "

                except Exception:
                    pass

    else:

        try:
            body = str(message.get_content())

        except Exception:
            body = ""

    return subject, body


# 4. Function to load emails from a folder
def load_emails(folder, label):

    emails = []

    # Check folder exists
    if not folder.exists():
        print("Folder not found:", folder)
        return emails

    # Read email files recursively
    for file_path in folder.rglob("*"):

        if not file_path.is_file():
            continue

        # Ignore README and non-email files
        if not re.match(r"^\d+", file_path.name):
            continue

        try:
            subject, body = extract_email(file_path)

            # Combine subject and body
            text = subject + " " + body

            if text.strip():

                emails.append({
                    "subject": subject,
                    "email_text": text,
                    "label": label,
                    "filename": file_path.name
                })

        except Exception:
            continue

    return emails


# 5. Load ham and spam emails
print("Loading legitimate emails...")

ham_emails = load_emails(ham_dir, 0)

print("Loading spam emails...")

spam_emails = load_emails(spam_dir, 1)


# 6. Combine both datasets
all_emails = ham_emails + spam_emails

df = pd.DataFrame(all_emails)


# 7. Display dataset information
print("\n========== DATASET INFORMATION ==========")

print("Total Emails:", len(df))

print("Legitimate Emails:", len(ham_emails))

print("Spam Emails:", len(spam_emails))


# 8. Display sample emails
print("\n========== SAMPLE EMAILS ==========")

print(df[["email_text", "label"]].head(5))


# 9. Check missing values
print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())


# 10. Save dataset as CSV
output_file = project_dir / "dataset" / "processed_emails.csv"

df.to_csv(output_file, index=False)

print("\nDataset saved successfully!")

print("Saved at:", output_file)
