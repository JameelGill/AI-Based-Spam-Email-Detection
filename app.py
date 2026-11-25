
import streamlit as st
import pandas as pd
import joblib
import re
import html

from pathlib import Path
from datetime import datetime

# Project paths
BASE_DIR = Path(r"D:\Spam_Email_Detection")

MODEL_FILE = BASE_DIR / "model" / "spam_model.pkl"
HISTORY_DIR = BASE_DIR / "history"
HISTORY_FILE = HISTORY_DIR / "prediction_history.csv"


# Clean email text (same preprocessing as training)
def clean_text(text):

    text = str(text).lower()
    text = html.unescape(text)

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


# Save prediction history
def save_history(email, result, confidence):

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    new_record = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Email": email,
        "Result": result,
        "Confidence": round(confidence * 100, 2)
    }])

    new_record.to_csv(
        HISTORY_FILE,
        mode="a",
        header=not HISTORY_FILE.exists(),
        index=False
    )


# Page settings
st.set_page_config(
    page_title="AI Spam Email Detector",
    page_icon="📧",
    layout="wide"
)

st.title("📧 AI-Based Spam Email Detection System")

st.write(
    "Detect spam and normal emails using Machine Learning."
)

st.divider()


# Check model
if not MODEL_FILE.exists():

    st.error("Trained model not found. Please train the model first.")
    st.stop()

model = load_model()


# Dashboard statistics
if HISTORY_FILE.exists():

    history = pd.read_csv(HISTORY_FILE)

else:

    history = pd.DataFrame(
        columns=["Date", "Email", "Result", "Confidence"]
    )

total = len(history)

spam_count = (
    history["Result"] == "Spam"
).sum() if total else 0

ham_count = (
    history["Result"] == "Ham"
).sum() if total else 0


col1, col2, col3 = st.columns(3)

col1.metric("Total Emails Checked", total)
col2.metric("Spam Emails", spam_count)
col3.metric("Normal Emails", ham_count)


st.divider()

st.subheader("🔍 Check Your Email")

with st.form("email_form"):

    email_text = st.text_area(
        "Paste your email here:",
        height=200,
        placeholder="Enter email subject and message..."
    )

    submit = st.form_submit_button(
        "Detect Email",
        use_container_width=True
    )


if submit:

    if not email_text.strip():

        st.warning("Please enter an email first.")

    else:

        cleaned_email = clean_text(email_text)

        if not cleaned_email:

            st.warning("Please enter valid email text.")

        else:

            prediction = model.predict([cleaned_email])[0]

            probabilities = model.predict_proba(
                [cleaned_email]
            )[0]

            confidence = max(probabilities)

            if prediction == 1:

                result = "Spam"

                st.error("🚨 This email is SPAM!")

            else:

                result = "Ham"

                st.success("✅ This email is NORMAL (HAM).")

            st.write(
                f"**Prediction Confidence:** {confidence * 100:.2f}%"
            )

            st.progress(float(confidence))

            save_history(
                email_text,
                result,
                confidence
            )

            st.info("Prediction saved in history.")


st.divider()

st.subheader("📊 Email Detection History")

if HISTORY_FILE.exists():

    history = pd.read_csv(HISTORY_FILE)

    st.dataframe(
        history.iloc[::-1],
        use_container_width=True
    )

    csv_data = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download History CSV",
        data=csv_data,
        file_name="spam_detection_history.csv",
        mime="text/csv"
    )

else:

    st.info("No email detection history available yet.")


st.caption("AI Spam Email Detection System | Machine Learning Project")
