## 🏆 **9th Place Solution: Augmentations + Modeling Tricks**

- **Ensemble**: Combined CatBoost and two LGBM Dart models.
- **Validation**: 8-fold GroupKFold for stable results.
- **Augmentations**:
  - **Flip**: Swap agents and adjust AdvantageP1 (used in TTA).
  - **Self-Play** and **Transitivity** explored but not used in final training.
- **Modeling**:
  - Subtracted AdvantageP1 from targets.
  - MultiRMSE objective in CatBoost.
- **Outcome**: Achieved strong leaderboard performance.

🔗 [Full Write-Up](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/549624)
