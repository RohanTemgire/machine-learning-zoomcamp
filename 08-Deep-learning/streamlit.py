import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
import os



# API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Fashion Classifier",
    layout="wide"
)


st.title("👗🧢 A Simple Fashion Classifier")

tab_info, tab_predict= st.tabs(["Info","🔍 Predict"])


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xception_v4_1_03_0.859.h5")
IMG_SIZE = (299, 299)

st.write("Current dir:", os.getcwd())
st.write("Files:", os.listdir(BASE_DIR))


classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

@st.cache_resource
def load_xception_model():
    return load_model(MODEL_PATH)

model = load_xception_model()


def predict_img(uploaded_file):
    # ------------------------
    # prediction starts here
    #----------------------
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(IMG_SIZE)

    x = np.array(img)
    X = np.expand_dims(x, axis=0)

    X = preprocess_input(X)

    preds = model.predict(X)

    class_index = int(np.argmax(preds))
    confidence = float(preds[0][class_index])

    return classes[class_index], confidence



with tab_info:
    st.markdown("""
    This application uses a **fine‑tuned Xception model** to classify fashion items into 10 distinct categories. The model has been trained on a curated dataset and optimized for accuracy across diverse clothing styles.

    ### Supported Classes
    - Dress  
    - Hat  
    - Longsleeve  
    - Outwear  
    - Pants  
    - Shirt  
    - Shoes  
    - Shorts  
    - Skirt  
    - T‑shirt  
    """)

with tab_predict:
    st.header("🔍 Predict image")

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", width=200)

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                label, confidence = predict_img(uploaded_file)

            st.success(f"Prediction: **{label}**")
            st.info(f"Confidence: **{confidence:.2f}**")
