import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

rf = pd.read_csv(
    "classic/checkpoints/A3_rf/oof_predictions.csv"
).sort_values("row_index")

xgb = pd.read_csv(
    "classic/checkpoints/A4_xgb/oof_predictions.csv"
).sort_values("row_index")

dl = np.load(
    "dl/checkpoints/dl_baseline_iter40/oof_predictions.npz"
)

rf_p = rf["probability"].to_numpy()
xgb_p = xgb["probability"].to_numpy()

y = rf["target"].to_numpy()

logits = dl["outputs"]

exp_logits = np.exp(
    logits - logits.max(axis=1, keepdims=True)
)

mlp_probs = exp_logits / exp_logits.sum(
    axis=1,
    keepdims=True
)

mlp_p = mlp_probs[:, 1]


assert np.array_equal(
    rf["target"].to_numpy(),
    xgb["target"].to_numpy()
)

assert np.array_equal(
    y,
    dl["labels"]
)

assert len(rf_p) == len(xgb_p) == len(mlp_p)


p_final = (
    rf_p +
    xgb_p +
    mlp_p
) / 3

pred = (p_final >= 0.5).astype(int)

oof_accuracy = accuracy_score(y, pred)

print(f"OOF accuracy: {oof_accuracy:.4f}")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=0xFACED,
)

fold_scores = []

for fold, (_, val_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
    score = accuracy_score(
        y[val_idx],
        pred[val_idx],
    )

    fold_scores.append(score)

    print(f"Fold {fold}: {score:.4f}")


print()
print(
    f"Ensemble CV: "
    f"{np.mean(fold_scores):.4f} ± "
    f"{np.std(fold_scores):.4f}"
)