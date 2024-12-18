## 📝 **BirdCLEF 2023**

- **Task**: Create a model to recognize bird species by their calls from audio data.

- **Rank**: 99th out of 1189 teams.

---

### 📂 **Training Notebook**

**Purpose**: Train a model to identify bird calls from audio recordings.

- **Key Elements**:
  - **Libraries**:
    - **TensorFlow**, **TensorFlow Addons**
    - **Librosa** for audio analysis.
    - **EfficientNet** for the CNN backbone.
    - **WandB** for experiment tracking.
  - **Data Processing**:
    - Converts raw audio to spectrograms.
    - Augmentation techniques: `TimeFreqMask`, `CutMix`, and `MixUp`.
  - **Model**:
    - **EfficientNet with Filter Stride Reduction (FSR)**.
    - Pretrained on BirdCLEF 2020, 2021, and 2022 datasets for transfer learning.
  - **Training Setup**:
    - Compatible with GPU, TPU, and TPU-VM.
    - Dynamic shape handling for spectrograms.
  - **Experimentation**:
    - Uses multiple spectrogram augmentation methods to enhance training data.

---

### 📂 **Inference Notebook**

**Purpose**: Run inference to predict bird species from audio recordings.

- **Key Elements**:
  - **Model**:
    - Pretrained **EfficientNet** model fine-tuned on BirdCLEF 2023.
  - **Data Processing**:
    - Converts raw audio to spectrograms during inference.
    - Processes 5-second audio clips for predictions.
  - **Hardware**:
    - Configures for automatic detection of GPU or TPU.
    - Uses dynamic shape inference for efficiency.
  - **Output**:
    - Predicts the probability of bird species presence for each recording.

---
