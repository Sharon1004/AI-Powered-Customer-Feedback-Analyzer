
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI-Powered Customer Feedback Analyzer",
    page_icon="💬",
    layout="centered"
)

st.title("💬 AI-Powered Customer Feedback Analyzer")
st.write(
    "Enter customer feedback and predict its sentiment, "
    "complaint category, and priority."
)

# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    sentiment_model = joblib.load(
        MODEL_DIR / "sentiment_pipeline.pkl"
    )

    complaint_model = joblib.load(
        MODEL_DIR / "complaint_model.pkl"
    )

    priority_model = tf.keras.models.load_model(
        MODEL_DIR / "priority_ann.keras"
    )

    vectorizer = joblib.load(
        MODEL_DIR / "priority_vectorizer.pkl"
    )

    label_encoder = joblib.load(
        MODEL_DIR / "priority_label_encoder.pkl"
    )

    preprocessor = joblib.load(
        MODEL_DIR / "priority_preprocessor.pkl"
    )

    return (
        sentiment_model,
        complaint_model,
        priority_model,
        vectorizer,
        label_encoder,
        preprocessor
    )


# --------------------------------------------------
# CHECK MODEL FILES
# --------------------------------------------------

required_files = [
    "sentiment_pipeline.pkl",
    "complaint_model.pkl",
    "priority_ann.keras",
    "priority_vectorizer.pkl",
    "priority_label_encoder.pkl",
    "priority_preprocessor.pkl"
]

missing_files = [
    file for file in required_files
    if not (MODEL_DIR / file).exists()
]

if missing_files:
    st.error(
        "Trained model files are missing. "
        "Please run train_models.py first."
    )

    for file in missing_files:
        st.write(f"- {file}")

    st.stop()


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("Enter Customer Details")

feedback_text = st.text_area(
    "Customer Feedback",
    placeholder="Example: The product quality is poor and delivery was delayed.",
    height=150
)

channel = st.selectbox(
    "Feedback Channel",
    ["App", "Website", "Email", "Phone", "Social Media"]
)

customer_type = st.selectbox(
    "Customer Type",
    ["New", "Regular", "Premium"]
)

order_value = st.number_input(
    "Order Value",
    min_value=0.0,
    value=1000.0,
    step=100.0
)

days_as_customer = st.number_input(
    "Days as Customer",
    min_value=0,
    value=30,
    step=1
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Analyze Feedback", type="primary"):

    if not feedback_text.strip():
        st.warning("Please enter customer feedback.")

    else:
        try:
            (
                sentiment_model,
                complaint_model,
                priority_model,
                vectorizer,
                label_encoder,
                preprocessor
            ) = load_models()

            # 1. SENTIMENT PREDICTION
            sentiment_prediction = sentiment_model.predict(
                [feedback_text]
            )[0]

            # 2. COMPLAINT TYPE PREDICTION
            complaint_prediction = complaint_model.predict(
                [feedback_text]
            )[0]

            # 3. PRIORITY PREDICTION
            text_features = vectorizer.transform(
                [feedback_text]
            ).toarray()

            metadata = pd.DataFrame({
                "channel": [channel],
                "customer_type": [customer_type],
                "order_value": [order_value],
                "days_as_customer": [days_as_customer]
            })

            metadata_features = preprocessor.transform(
                metadata
            )

            if hasattr(metadata_features, "toarray"):
                metadata_features = metadata_features.toarray()

            priority_features = np.concatenate(
                [text_features, metadata_features],
                axis=1
            )

            priority_probabilities = priority_model.predict(
                priority_features,
                verbose=0
            )

            priority_index = np.argmax(
                priority_probabilities,
                axis=1
            )[0]

            priority_prediction = label_encoder.inverse_transform(
                [priority_index]
            )[0]

            priority_confidence = float(
                np.max(priority_probabilities)
            )

            # --------------------------------------------------
            # DISPLAY RESULTS
            # --------------------------------------------------

            st.success("Feedback Analysis Completed!")

            st.subheader("Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Sentiment",
                    str(sentiment_prediction)
                )

            with col2:
                st.metric(
                    "Complaint Type",
                    str(complaint_prediction)
                )

            with col3:
                st.metric(
                    "Priority",
                    str(priority_prediction)
                )

            st.write(
                f"**Priority Confidence:** "
                f"{priority_confidence:.2%}"
            )

            st.divider()

            st.subheader("Submitted Feedback")
            st.write(feedback_text)

        except Exception as e:
            st.error(f"Prediction error: {e}")
