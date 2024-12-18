## 📝 **Predict Student Performance from Game Play**

- **Task**: Develop a model to predict student performance during game-based learning in real-time using game log data.

- **Rank**: 276th out of 2051 teams.

---

### 📂 **Training & Inference Notebook**

**Purpose**: Train a model and perform inference to predict student performance based on game log data.

### **Pipeline**:
   - Load training data and labels from CSV files.
   - Data types optimized for memory efficiency using `numpy` data types.
   - Compute `delt_time` as the difference in elapsed time between consecutive events.
   - Clip large time differences to prevent outliers.
   - Create features based on `session_id`, `event_name`, `level`, `hover_duration`, and coordinates (`room_coor_x`, `room_coor_y`).
   - **CatBoostClassifier** and **XGBClassifier** for classification tasks.
   - Grouped cross-validation using `GroupKFold` to ensure sessions are not split across folds.
   - Train models on engineered features.
   - Optimize hyperparameters and evaluate using **F1 Score**.
   - Perform predictions on test data.
   - Save the final predictions for submission.

### **Team Members**:
- [@mohammad2012191](https://www.kaggle.com/mohammad2012191)
- [@aliibrahimali](https://www.kaggle.com/aliibrahimali)
