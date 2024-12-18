## 📝 **HMS - Harmful Brain Activity Classification**

- **Task**: Develop a model trained on electroencephalography (EEG) signals recorded from critically ill hospital patients to classify seizures and other types of harmful brain activity.

---

### 📂 **Pipeline**

1. **Data Preprocessing**:
   - **Spectrogram Generation**:
     - Each EEG signal is converted into **8 spectrograms**, representing different brain regions.
     - Each spectrogram is of size **128x256**.
   - **Image Formation**:
     - The 8 spectrograms are **concatenated** to form a **single large image** (e.g., a composite of 512x512).
     - Alternatively, spectrograms are stacked along the **depth channel** to create a **2.5D image**.

2. **Model Architecture**:
   - **EfficientNet Variants**: B0, B1, B2, B3, B4.
   - **MobileNet** for lightweight inference.
   - Input size: 512x512 or the resized composite spectrogram image.

3. **Training Configuration**:
   - **Cross-Validation**:
     - **Train** on all samples.
     - **Validate** only on samples where the number of votes is between **3 and 20**.
   - **Loss Function**: **Kullback-Leibler Divergence (KLD)** to handle the probabilistic nature of votes.
   - **Augmentation**: Applied using **Albumentations**:
     - Random rotations, scaling, flipping, color jitter, and Gaussian noise.

