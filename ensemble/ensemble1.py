import pandas as pd


lr = pd.read_csv("a1_lr.csv")
rf = pd.read_csv("a3_rf.csv")
xgb = pd.read_csv("a4_xgb.csv")

assert lr["PassengerId"].equals(rf["PassengerId"])
assert lr["PassengerId"].equals(xgb["PassengerId"])


pred = (
    (
        lr["Survived"].to_numpy()
        + rf["Survived"].to_numpy()
        + xgb["Survived"].to_numpy()
    ) >= 2
).astype(int)


submission = pd.DataFrame({
    "PassengerId": lr["PassengerId"],
    "Survived": pred,
})


submission.to_csv(
    "ensemble_A1_RF_XGB.csv",
    index=False
)

print("Saved: ensemble_A1_RF_XGB.csv")