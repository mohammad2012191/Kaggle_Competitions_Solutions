## 📝 **Learning Agency Lab - Automated Essay Scoring 2.0**

- **Task**: Train a model to score student essays automatically.

- **Rank**: 188th out of 2706 teams.

---

**Purpose**: Perform inference to score student essays using a trained NLP deep learning model.

- **Key Elements**:
  - **Libraries**:
    - **PyTorch**, **PyTorch Lightning** for model handling and training.
    - **Transformers** for tokenization and pre-trained models.
    - **Scikit-learn** for metrics and preprocessing.

  - **Pipeline Steps**:
    1. **Directory Settings**: Path configurations for input and output data.
    2. **Data Loading**: Reads the test dataset.
    3. **Tokenizer**: Uses `AutoTokenizer` for text tokenization.
    4. **Dataset**: Custom `Dataset` class for preparing inputs.
    5. **Model Definition**: DebertaV3 model with mean pooling for essay scoring.
    6. **Inference**: Loads the trained model and generates predictions.
  - **Utilities**:
    - Functions for tokenization, data loading, and prediction formatting.
    - Inference results are prepared for submission.
