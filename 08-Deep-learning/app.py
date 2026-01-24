from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from model_utils import predict_img


app = FastAPI(title="Fashion Classifier")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}



@app.get("/")
def home():
    return {"message": "Fashion Classifier API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    _, ext = os.path.splitext(file.filename.lower())

    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid content type.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict_img(file_path)
        return result
    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))