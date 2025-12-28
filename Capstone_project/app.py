from fastapi import FastAPI, UploadFile, File
import shutil
import os
from model_utils import train_model, test_model, predict_image

app = FastAPI(title="Tumor Detection API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Tumor Detection API is running"}

# ------------------
# PREDICT
# ------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(file_path)
    return result

# ------------------
# TRAIN
# ------------------
@app.post("/train")
def train(epochs: int = 5):
    return train_model(epochs)

# ------------------
# TEST
# ------------------
@app.get("/test")
def test():
    return test_model()
