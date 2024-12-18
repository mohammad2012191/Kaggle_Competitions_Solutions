## 📝 **Novozymes Enzyme Stability Prediction**

- **Task**: Develop a model that can predict the ranking of protein thermostability (measured by melting point, `tm`) after single-point amino acid mutation and deletion.

- **Rank**: 290th out of 2483 teams.

---

### 📂 **Training Pipeline**

1. **ThermoNet Model**:
   - **Architecture**: 3D Convolutional Neural Network (CNN) inspired by **ThermoNet**.
   - **Training Data**: 
     - Largest dataset of mutations (~14,656 unique mutations).
     - Focused on **destabilizing mutations** (`ΔΔG < 0`).
   - **Features**: Voxel features derived from protein structures.

2. **ProtBERT Model**:
   - **Architecture**: Transformer-based model using **ProtBERT** or **ESM2** for protein sequences.
   - **Input**: Wild-type and mutated protein sequences with mutation location.

3. **pLDDT Predictions**:
   - **Approach**:
     - Extracts pLDDT values from wild-type and mutated protein structures.
     - Computes differences between wild-type and mutant pLDDT scores.
   
4. **Ensembling**:
   - Combines predictions from:
     - **ThermoNet**
     - **ProtBERT**
     - **pLDDT Difference Method**
     - ...Other models.
   - Uses rank-based ensembling to improve stability and accuracy.

---

### **Team Members**:
- [@ksouriazer](https://www.kaggle.com/ksouriazer)
- [@ihebch](https://www.kaggle.com/ihebch)
- [@mohammad2012191](https://www.kaggle.com/mohammad2012191)
