## 📝 **Feedback Prize - English Language Learning**

- **Task**: Develop a model to assess the language proficiency of 8th-12th grade English Language Learners (ELLs).

- **Rank**: 223rd out of 2654 teams.

---

### 📂 **Training Notebook**

**Purpose**: Train Transformer-based models to assess various aspects of English proficiency.

### **Pipeline**:
   - Load training data containing essays and proficiency scores.
   - Targets include: `cohesion`, `syntax`, `vocabulary`, `phraseology`, `grammar`, `conventions`.
   - Perform 4-fold **MultiLabelStratifiedKFold** to validate across multiple targets.
   - Apply padding and truncation to a maximum length of 512 tokens.
   - Use **DeBERTa-v3-base/large** Transformer model.
   - Apply mean pooling to aggregate hidden states for final predictions.
   - Train the model using PyTorch Lightning.
   - Implement **gradient clipping** for stability.
   
---

### 📂 **Inference Notebook**

**Purpose**: Perform inference using multiple trained models to predict English proficiency scores.

- **Configurations**:
  - **Models Used**:
    - **CFG1**: `deberta-v3-base` – 10-fold CV/LB: 0.4595/0.44
    - **CFG2**: `deberta-v3-large` – 10-fold CV/LB: 0.4553/0.44
    - **CFG3**: `deberta-v2-xlarge` – 10-fold CV/LB: 0.4604/0.44
    - ...Additional variations with some unscaled versions.

- **Inference Pipeline**:
  - **Tokenizer**: `AutoTokenizer` from HuggingFace.
  - **Ensembling**:
    - Combines predictions from multiple configurations for robust results.
  - **Output**:
    - Predicts scores for six language proficiency categories.

---
