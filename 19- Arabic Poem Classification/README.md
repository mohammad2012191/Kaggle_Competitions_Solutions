## 📝 **Arabic Poem Classification**

- **Task**: Develop a classification model that accurately categorizes Arabic poems based on their era.

---

### 📂 **Training Pipeline**

1. **Data Preprocessing**:
   - **Duplicate Removal**:
     - Identified and removed duplicates in the training data.
     - Found significant overlap between train and test data, allowing 80% accuracy through direct mapping.
   - **Text Concatenation**:
     - Combined `Title`, `Author`, and `Poem` into a single text string.
   - **Tokenization**:
     - Used the **CAMeL-Lab/bert-base-arabic-camelbert-ca** tokenizer.
     - Limited input to the **first 256 tokens** for optimal performance.

2. **Cross-Validation**:
   - **StratifiedGroupKFold**:
     - Stratified by the target categories.
     - Grouped by authors to ensure no author appears in both train and validation folds.

3. **Model**:
   - **DeBERTa Variant**: Fine-tuned on classic Arabic text.
   - **Custom Training**:
     - Implemented with **PyTorch Lightning**.
     - Included techniques like:
       - **Reinitializing the last layer**.
       - **Freezing layers**.
       - **Custom pooling layers**.
       - **Custom loss functions**.

4. **Training Configuration**:
   - **Epochs**: 5-fold training loop taking ~2 hours on **Nvidia A4000 16GB** GPU.
   - **Batch Size**: Configured for GPU memory efficiency.
   - **Optimizer**: **AdamW**.
   - **Learning Rate Scheduling**: Applied cosine learning rate schedule with warmup.

---

### **Key Insights**:
- **Data Leakage**:
  - Discovered significant overlap between train and test data (~80%).
  - Achieved high accuracy by mapping duplicates directly.
- **Diacritics**:
  - Identified that diacritics (تشكيل) significantly influenced classification results.
- **Model Performance**:
  - Achieved **CV 80%** and **LB 82%** without relying on duplicates.

---
