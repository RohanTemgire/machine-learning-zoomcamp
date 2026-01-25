# Brain Tumor Classification using EfficientNetB2

## Overview
This project implements a deep learning–based medical image classification system for detecting and classifying brain tumors from MRI images. The model classifies scans into four categories: **glioma tumor**, **meningioma tumor**, **pituitary tumor**, and **no tumor**.

The focus of this project is not only on model training, but on building a **medically realistic, rigorously evaluated pipeline**, emphasizing correct metrics, error analysis, and deployment readiness.

---

## Dataset
- **Modality:** Brain MRI images
- **Classes:**  
  - Glioma tumor  
  - Meningioma tumor  
  - Pituitary tumor  
  - No tumor  
- **Training samples:** ~2,870  
- **Test samples:** ~394  

Given the limited dataset size and subtle inter-class differences, this project emphasizes **honest evaluation and limitations analysis** rather than over-optimistic accuracy claims.

---

## Model Architecture
- **Backbone:** EfficientNetB2 (ImageNet pretrained)
- **Input size:** 260 × 260 × 3
- **Classifier head:**
  - Global Average Pooling
  - Batch Normalization
  - Dense (ReLU)
  - Dropout
  - Softmax output (4 classes)

### Training Strategy
1. **Stage 1 – Feature Extraction**
   - Backbone frozen
   - Classifier head trained from scratch
2. **Stage 2 – Controlled Fine-Tuning**
   - Last few backbone layers unfrozen
   - Very low learning rate to avoid feature drift

---

## Data Handling
- **Train / Validation / Test split**
- **Medical-safe augmentation** applied only to training data:
  - Small rotations
  - Minor shifts
  - Zoom
  - Horizontal flips
- **No augmentation** applied to validation or test sets
- ImageNet preprocessing used consistently

---

## Evaluation Metrics
Due to the medical nature of the task, **accuracy alone was not used**.

Reported metrics include:
- Per-class **Precision, Recall, and F1-score**
- **Macro F1-score** (treats all tumor types equally)
- **Weighted F1-score**
- **Confusion Matrix** for detailed error analysis

This revealed important failure modes, particularly confusion between visually similar tumor types.

---

## Failure Analysis
To better understand model limitations, the most confident incorrect predictions were extracted and visually inspected. This analysis showed:
- Significant visual overlap between glioma and meningioma tumors
- Small or diffuse tumors that are difficult to classify from whole-image inputs
- Some cases that are ambiguous even for human interpretation

This step provided crucial insight into why performance plateaus with limited data.

---

## Model Optimization
To demonstrate deployment readiness:
- The trained model was **quantized to INT8** using TensorFlow Lite
- Model size was significantly reduced
- Inference speed was benchmarked on CPU

This shows the feasibility of running the model in resource-constrained environments.

---

## Results (Summary)
- Multi-class classification on a small medical dataset
- Performance limited by dataset size and lack of localization
- Metrics and error analysis confirm the model learns meaningful but incomplete representations

Rather than overfitting or inflating results, this project prioritizes **reproducibility, transparency, and clinical realism**.

---

## Limitations
- Limited dataset size per tumor class
- Whole-image classification without explicit tumor localization
- High inter-class similarity between certain tumor types
- No multimodal clinical data (e.g., patient metadata)

---

## Future Work
- Tumor localization or segmentation-based pipelines
- Patch-based classification
- Binary (tumor vs no tumor) followed by tumor subtype classification
- Larger and more diverse datasets
- Integration of explainability methods once performance stabilizes

---

## Conclusion
This project demonstrates an end-to-end medical imaging pipeline with strong emphasis on correct evaluation, failure analysis, and deployment considerations. While constrained by data availability, it reflects real-world challenges in medical AI and provides a solid foundation for future improvements.
