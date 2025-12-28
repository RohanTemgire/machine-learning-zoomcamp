import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

# ------------------
# CONSTANTS
# ------------------
MODEL_PATH = "efficentnetmodels/efficientnet_best_recall_34_0.951.keras"
TRAIN_DIR = "data/Training"
TEST_DIR = "data/Testing"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]

# ------------------
# LOAD MODEL
# ------------------
def load_trained_model():
    return load_model(MODEL_PATH)

# ------------------
# TRAIN MODEL
# ------------------
def train_model(epochs=5):
    train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Keeps data in 0-255 range as EfficientNet expects
    rotation_range=40,        # Rotate images to simulate different head positions
    shear_range=0.2,          # Distort image angle
    zoom_range=0.2,           # Zoom in (simulate different scan proximities)
    vertical_flip=True,       # Flips upside down (Valid for Tumors, bad for Fashion)
    fill_mode='nearest',      # How to fill missing pixels after rotation
    validation_split=0.2      # Use 20% of data for validation
    )


    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    model = load_trained_model()

    model.fit(
        train_generator,
        epochs=epochs
    )

    model.save(MODEL_PATH)

    return {"status": "training completed", "epochs": epochs}

# ------------------
# TEST / EVALUATE
# ------------------
def test_model():
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    model = load_trained_model()

    preds = model.predict(test_generator)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_generator.classes

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )

    matrix = confusion_matrix(y_true, y_pred).tolist()

    return {
        "classification_report": report,
        "confusion_matrix": matrix
    }

# ------------------
# PREDICT SINGLE IMAGE
# ------------------
def predict_image(img_path):
    model = load_trained_model()

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)[0]
    class_index = np.argmax(preds)

    return {
        "prediction": CLASS_NAMES[class_index],
        "confidence": float(preds[class_index])
    }
