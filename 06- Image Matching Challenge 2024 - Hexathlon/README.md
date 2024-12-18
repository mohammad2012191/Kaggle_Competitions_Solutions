## 📝 **Image Matching Challenge 2024 - Hexathlon**

- **Task**: Construct precise 3D maps using images from diverse environments.

- **Rank**: 71st out of 929 teams.

### **Approach**:

- **Feature Detection and Matching**:
  - Combined multiple keypoint detectors and descriptors:
    - **KeyNet**, **GFTT**, **DoG**, and **Harris Corner Detector**.
  - Matched features between image pairs for robust correspondence.

- **Robust Estimation**:
  - Applied **RANSAC** to estimate the fundamental matrix and remove outliers.
  
- **3D Reconstruction**:
  - Utilized **Pycolmap** to create 3D models from matched image pairs.
