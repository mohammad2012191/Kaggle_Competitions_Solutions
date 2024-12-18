## 📝 **19th Place Solution: LMSYS - Chatbot Arena Human Preference Predictions**

- **Models Used**:
  - **Gemma-2-9B-it**
  - Experimented with DeBERTa, LLaMA 3.1, Phi,...etc

- **Techniques**:
  - **LoRA Config**: Added `o_proj` and `gate_proj`.
  - **Max Length**: Increased to 3072.
  - **Frozen Layers**: Set to 0 for better results.
  - **Custom Classification Head** Gives better results.
  - **External Data**: Combined competition data with 21k additional samples.

- **TTA (Test-Time Augmentation)**:
  - Used augmented inputs by flipping responses.

🔗 [Full Write-Up](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527704)
