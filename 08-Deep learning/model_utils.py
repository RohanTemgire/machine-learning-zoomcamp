import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input


MODEL_PATH = 'xception_v4_1_03_0.859.h5'
IMG_SIZE = (299, 299)

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


model = load_model('xception_v4_1_03_0.859.h5')

def predict_img(img_path):

    img = load_img(img_path, target_size = IMG_SIZE)
    
    x = np.array(img)
    X = np.expand_dims(x, axis=0)  # cleaner than np.array([x])

    X = preprocess_input(X)

    preds = model.predict(X)

    class_index = int(np.argmax(preds))
    confidence = float(preds[0][class_index])

    return {
        "prediction": classes[class_index],
        "confidence": confidence
    }
