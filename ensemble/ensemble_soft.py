import numpy as np
import pandas as pd

rf = pd.read_csv("a3_rf_proba.csv")
xgb = pd.read_csv("a4_xgb_proba.csv")
mlp_p = np.load("a6_mlp_proba.npy")

assert rf["PassengerId"].equals(xgb["PassengerId"])
assert len(rf) == len(mlp_p)

rf_p = rf["Probability"].to_numpy()
xgb_p = xgb["Probability"].to_numpy()

# Final soft-voting setup: equal weights.
rf_weight = 1 / 3
xgb_weight = 1 / 3
mlp_weight = 1 / 3

p_final = rf_weight * rf_p + xgb_weight * xgb_p + mlp_weight * mlp_p
pred = (p_final >= 0.5).astype(int)

submission = pd.DataFrame({
    "PassengerId": rf["PassengerId"],
    "Survived": pred,
})

submission.to_csv("ensemble_RF_XGB_MLP_soft.csv", index=False)
print("Saved: ensemble_RF_XGB_MLP_soft.csv")
