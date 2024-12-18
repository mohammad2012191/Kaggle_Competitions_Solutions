## 📝 **RSNA 2024 Lumbar Spine Degenerative Classification**

- **Task**: Develop a model to detect and classify degenerative spine conditions using lumbar spine MR images.

- **Rank**: 104th out of 1874 teams.

---

**Purpose**: Perform inference using an ensemble of YOLO models and a 3D Vision Transformer.

- **Ensemble Components**:
  - **YOLO Models**:
    - Detect and classify degenerative conditions using object detection.
  - **3D Vision Transformer**:
    - Single-stage classifier for 3D MR image data.

- **Key Steps**:
  - **Data Loading**:
    - Reads DICOM files and converts them to images.
    - Normalizes pixel values for consistency.
  - **Model Predictions**:
    - Loads YOLO and MaxViT models and applies them to detect spinal abnormalities.
    - Combines predictions from multiple models for robust results.
  - **Post-processing**:
    - Aggregates predictions to determine severity and classification labels.

---
