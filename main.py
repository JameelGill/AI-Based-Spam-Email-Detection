
import joblib
import os
import re
import html

# Function to clean email text
def clean_text(text):
    text = str(text)
    text = text.lower()

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Load trained model
model_path = os.path.join("model", "spam_model.pkl")

if not os.path.exists(model_path):
    print("Error: Trained model not found!")
    exit()

model = joblib.load(model_path)

print("=" * 45)
print("       AI SPAM EMAIL DETECTION SYSTEM")
print("=" * 45)

while True:

    print("\nEnter your email text below.")
    print("Type 'exit' to close the system.")

    email = input("\nEmail: ").strip()

    if email.lower() == "exit":
        print("\nThank you for using the system!")
        break

    if not email:
        print("Please enter some email text.")
        continue

    # Clean email before prediction
    cleaned_email = clean_text(email)

    if not cleaned_email:
        print("Please enter valid email text.")
        continue

    # Predict email
    prediction = model.predict([cleaned_email])[0]

    print("\nPrediction Result:")

    if prediction == 1:
        print("WARNING! This is a SPAM email.")
    else:
        print("This is a HAM (Normal) email.")

    print("-" * 45)
