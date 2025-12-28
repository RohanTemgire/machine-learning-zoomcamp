# 🧠 Brain Tumor Detection System (FastAPI + Streamlit)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)

## 1. About the Project

This project is an **end-to-end Brain Tumor Detection system** built using **Deep Learning**, **FastAPI**, **Streamlit**, and **Docker**.

The model classifies brain MRI images into **four categories**:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

The system provides:
- A **REST API** (FastAPI) for prediction, training, and testing
- A **web-based GUI** (Streamlit) for easy interaction
- **Dockerized deployment** for easy setup and portability

⚠️ **Disclaimer:**  
This project is for **educational and research purposes only** and **must not be used for medical diagnosis**.

---

## 2. Dataset

The dataset consists of **MRI images of brain tumors** organized into training and testing folders.

### 📥 Download Dataset
You can download the dataset directly from this GitHub repository:
### 📂 Directory Structure
After extraction, ensure your `data/` folder looks exactly like this:

```text
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## 3. Model Architecture
- Base Model: EfficientNetB3 (Pre-trained on ImageNet)
- Input Shape: (224, 224, 3)
- Output Layer: Softmax (4 Neurons)
- Loss Function: Categorical Crossentropy
- Optimizer: Adam

Why EfficientNet? We chose EfficientNetB3 because it offers a superior balance between accuracy and computational efficiency compared to older models like ResNet or VGG. It uses compound scaling to optimize depth, width, and resolution.

The trained model is stored in:
    efficientnetmodels/efficientnet_best_recall_*.keras

## 4. 🚀 How to Run (Local Installation)

1. Step 1: Clone the Repository
- git clone [https://github.com/](https://github.com/)<your-username>/<repo-name>.git 
- cd Capstone_project

2. Step 2: Install Dependencies
- pip install -r requirements.txt

3. Step 3: Start the Backend (FastAPI)
- uvicorn app:app --reload

4. Step 4: Start the Frontend (Streamlit)
- streamlit run streamlit_app.py

## 5. 🐳 Docker Support (Recommended)

Skip the manual installation and run everything in a container.

### Option A: Build from Source

```bash
# 1. Build the image
docker build -t brain-tumor-detection .

# 2. Run the container
docker run -p 8000:8000 -p 8501:8501 brain-tumor-detection
```
### Option B: Access
Once running, access the services at:

Frontend: http://localhost:8501

Backend Docs: http://localhost:8000/docs

## 6. Using the Application
### 🔍 Predict
1. Navigate to the Prediction tab in Streamlit.

2. Upload a .jpg, .png, or .jpeg MRI scan.

3. The model will output the Predicted Class and a Confidence Score (e.g., "Meningioma (98.5%)").

### 🏋️ Train
1. Go to the Train tab.

2. Set the number of epochs (e.g., 10 or 20).

3. Click Start Training.

4. The backend triggers a background task.

### 🧪 Evaluate
1. Go to the Evaluation tab.

2. Click Evaluate Model.

3. The system generates a classification report (Precision, Recall, F1-Score) and plots a Confusion Matrix to visualize misclassifications.