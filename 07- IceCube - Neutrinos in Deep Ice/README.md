## 📝 **IceCube - Neutrinos in Deep Ice**

- **Task**: Develop a model to predict a neutrino particle’s direction based on data from the IceCube detector at the South Pole.

- **Rank**: 78th out of 812 teams.

---

### 📂 **Preprocessing Notebook**

**Purpose**: Prepare training data by extracting and processing relevant features.

- **Key Steps**:
  - **Feature Selection**: 
    - Max 96 pulses per event.
    - 7 features: `time`, `charge`, `aux`, `x`, `y`, `z`, `rank`.
  - **Sensor Geometry**: Load sensor coordinates from `sensor_geometry.csv`.
  - **Event Processing**:
    - Rank pulses by importance based on charge and validity.
    - Filter out insignificant pulses.
  - **Output**: Saves processed data batches in `.npz` format.

---

### 📂 **Training Notebook**

**Purpose**: Train an LSTM model to predict neutrino directions.

- **Model Configuration**:
  - **LSTM Units**: 192.
  - **Inputs**: 96 pulses, 6 features.
  - **Loss Function**: Custom angular distance score.
  - **Hardware**: Configured for TPU/GPU.
    
---

### 📂 **Inference Notebook**

**Purpose**: Generate predictions using the trained models.

- **Steps**:
  - **Model Loading**: Load an ensemble of 5 models (trained at different epochs).
  - **Ensemble Weights**: Combine predictions with predefined weights.
  - **Input**: Load test batches in `.parquet` format.
  - **Output**: Predict neutrino directions (azimuth and zenith angles).

---
