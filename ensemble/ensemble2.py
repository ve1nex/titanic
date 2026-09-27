import numpy as np
import pandas as pd

rf = pd.read_csv("a3_rf.csv")
xgb = pd.read_csv("a4_xgb.csv")
mlp = np.load("finalMLP.npy").astype(int)

assert rf["PassengerId"].equals(xgb["PassengerId"])
assert len(rf) == len(mlp)

rf_pred = rf["Survived"].to_numpy()
xgb_pred = xgb["Survived"].to_numpy()

pred = (
    (rf_pred + xgb_pred + mlp) >= 2
).astype(int)


submission = pd.DataFrame({
    "PassengerId": rf["PassengerId"],
    "Survived": pred,
})

submission.to_csv(
    "ensemble_RF_XGB_MLP.csv",
    index=False
)

print("Saved: ensemble_RF_XGB_MLP.csv")