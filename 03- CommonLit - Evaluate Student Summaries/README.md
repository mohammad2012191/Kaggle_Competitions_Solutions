## 📝 **27th Place Solution: CommonLit - Evaluate Student Summaries**

- **Approach**:
  - **Ensemble of Models**: First-stage models with DeBERTa-v3-large backbone.
  - **Custom Pooling**: Added prompts to input for diversity.
  - **Second Stage**: Stacking model (LGBM, XGBoost, CatBoost) using first-stage predictions.

- **Techniques**:
  - Used FB3 labels and metadata features (e.g., grade).
  - Experimented with Bi-LSTM for contextual understanding.

- **Team Members**: 
  - [@muhammad4hmed](https://www.kaggle.com/muhammad4hmed)
  - [@mohammad2012191](https://www.kaggle.com/mohammad2012191)
  - [@cody11null](https://www.kaggle.com/cody11null)
  - [@ivanisaev](https://www.kaggle.com/ivanisaev)
  - [@ihebch](https://www.kaggle.com/ihebch)

🔗 [Full Write-Up](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446542)
