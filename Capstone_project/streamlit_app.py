import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide"
)

st.title("🧠 Brain Tumor Detection System")
st.caption("For educational purposes only. Not for medical diagnosis.")

# ----------------------------
# TABS
# ----------------------------
tab_predict, tab_train, tab_test = st.tabs(["🔍 Predict", "🏋️ Train", "🧪 Test"])

# =====================================================
# 🔍 PREDICT TAB
# =====================================================
with tab_predict:
    st.header("Predict Tumor Type")

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        st.image(
            uploaded_file,
            caption="Uploaded Image",
            width="stretch"
        )

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    files={"file": uploaded_file}
                )

            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: **{result['prediction']}**")
                st.info(f"Confidence: **{result['confidence']:.2f}**")
            else:
                st.error("Prediction failed")

# =====================================================
# 🏋️ TRAIN TAB
# =====================================================
with tab_train:
    st.header("Train Model")

    st.warning(
        "Training may take several minutes and will use your system resources."
    )

    epochs = st.number_input(
        "Number of epochs",
        min_value=1,
        max_value=50,
        value=5,
        step=1
    )

    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            response = requests.post(
                f"{API_BASE_URL}/train",
                params={"epochs": epochs}
            )

        if response.status_code == 200:
            st.success("Training completed successfully!")
            st.json(response.json())
        else:
            st.error("Training failed")

# =====================================================
# 🧪 TEST TAB
# =====================================================
with tab_test:
    st.header("Evaluate Model")

    if st.button("Run Evaluation"):
        with st.spinner("Evaluating model..."):
            response = requests.get(f"{API_BASE_URL}/test")

        if response.status_code == 200:
            results = response.json()

            st.subheader("Classification Report")
            report_df = pd.DataFrame(results["classification_report"]).transpose()
            st.dataframe(report_df)

            st.subheader("Confusion Matrix")
            cm_df = pd.DataFrame(
                results["confusion_matrix"],
                columns=report_df.index[:-3],
                index=report_df.index[:-3]
            )
            st.dataframe(cm_df)

        else:
            st.error("Evaluation failed")

