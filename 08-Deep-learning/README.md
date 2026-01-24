# 👗🧢 Fashion Image Classification with Xception

🚀 **Live Demo:** https://fashion-classifier-1.streamlit.app/

A deep learning–based fashion image classifier built using **TensorFlow / Keras** and deployed using **Streamlit**.  
The model classifies clothing images into **10 fashion categories** with ~**90% test accuracy**.

This project demonstrates an end-to-end ML workflow including **dataset preparation**, **transfer learning**, **model fine-tuning**, and **deployment**.

---

## 🔍 Overview

- **Model**: Fine-tuned Xception (transfer learning + custom training)
- **Task**: Multi-class image classification
- **Classes**: 10 fashion categories
- **Frameworks**: TensorFlow, Keras, Streamlit
- **Deployment**: Streamlit (current), FastAPI backend included

---

## 🧠 Model & Approach

The model is based on **Xception**, a deep convolutional neural network pre-trained on ImageNet.

### Training Strategy
1. Loaded Xception with ImageNet weights
2. Replaced the classifier head
3. Fine-tuned the network on a fashion dataset
4. Optimized for accuracy on unseen test data

### Performance
- **Test accuracy**: ~**0.89 (≈ 90%)**
- Evaluated on a held-out test set

---

## 📂 Dataset

The model was trained on the **Clothing Dataset (Small)**:

```bash
git clone git@github.com:alexeygrigorev/clothing-dataset-small
```

### Classes

- Dress
- Hat
- Longsleeve
- Outwear
- Pants
- Shirt
- Shoes
- Shorts
- Skirt
- T-shirt

---

## 🖥️ Streamlit Application

The Streamlit app allows users to:

- Upload an image (`.jpg`, `.jpeg`, `.png`)
- Preview the uploaded image
- Get a predicted clothing category with a confidence score

### Features

- Model loaded once using caching for performance
- Clean UI with tab-based layout
- Real-time predictions

---

## ⚙️ FastAPI Backend (Included)

This repository also contains a **FastAPI backend** with `/predict` endpoints that:

- Accept image uploads
- Run model inference
- Return JSON predictions

> **Note**  
> The FastAPI backend is included for demonstration and extensibility.  
> The project is currently deployed using **Streamlit only** for simplicity.
