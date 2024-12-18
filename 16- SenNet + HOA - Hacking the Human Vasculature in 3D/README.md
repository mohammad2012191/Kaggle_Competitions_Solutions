## 📝 **SenNet + HOA - Hacking the Human Vasculature in 3D**

- **Task**: Create a model trained on 3D Hierarchical Phase-Contrast Tomography (HiP-CT) data from human kidneys to segment blood vessels.

---

### 📂 **Training Pipeline**
   - **U-Net** with a **SE-ResNeXt50** backbone from `segmentation_models_pytorch`.
   - **2.5D Approach**: Combines 2D slices from 3D volumes to leverage 2D segmentation models.
   - Segment large 3D images using a tiling approach
   - Loss Function: **Dice Loss** for segmentation tasks.
   - **Validation**: Performed using fold-based cross-validation (split by kidney datasets).
   - Applied various data augmentations using **Albumentations** to improve generalization:
     - **Rotation**: Random rotations up to 270°.
     - **Scaling**: Random scaling within the range (0.8, 1.25).
     - **Random Cropping**: 512x512 crops.
     - **Random Brightness/Contrast**.
     - **Gaussian Blur** and **Motion Blur**.
     - **Grid Distortion** for spatial augmentations.

---

### 📂 **Inference Pipeline**

   - Ensemble of models trained on different resolutions (512x512 and 1024x1024).
   - Segment large 3D images using a tiling approach (e.g., 1024x1024 tiles with 75% overlap).
   - **Stride**: Tiles overlap to ensure no information loss.
   - **Thresholding**: Apply percentile-based thresholds to refine segmentation masks.
