## 📝 **RSNA 2022 Cervical Spine Fracture Detection**

- **Task**: Develop a model to detect and localize fractures in cervical spine vertebrae from medical images.

- **Rank**: 100th out of 883 teams.

---

### 📂 **Training Notebook**

**Purpose**: Train an EfficientNet-based model to detect cervical spine fractures.

- **Key Elements**:
  - **Libraries**:
    - **PyTorch**, **Torchvision**, **pydicom** for DICOM image processing.
  - **Model**:
    - **EfficientNetV2-S** pretrained on ImageNet.
    - The classifier predicts fractures for vertebrae C1 to C7.
  - **Training Setup**:
    - Uses **GroupKFold** for cross-validation.
    - **Binary Cross-Entropy Loss** for multi-label classification.
    - Weighted loss to emphasize fracture detection.
  - **Data**:
    - Images preprocessed and resized to `384x384` tensors.
  - **Tracking**: 
    - **Weights & Biases (WandB)** for logging training progress.

---

### 📂 **Inference Notebook**

**Purpose**: Perform inference using the trained model to predict cervical spine fractures.

- **Key Elements**:
  - **Model**:
    - Ensemble of 5 **EfficientNetV2-S** models trained on different folds.
  - **Prediction Logic**:
    - Generates probabilities for fractures and vertebrae presence (C1-C7).
    - Aggregates predictions using weighted averages.
    - Calculates the overall fracture probability for each patient.
  - **Data**:
    - Loads test images in `.dcm` (DICOM) format.
    - Images resized to `384x384` tensors.
  - **Output**:
    - Fracture predictions for each vertebra and overall patient fracture probability.

---
