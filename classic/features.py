import pandas as pd


def feature_engineering(df: pd.DataFrame, config) -> pd.DataFrame:
    """Extract the title retained by the Classic ML experiments, then drop raw names."""
    if not config.feature_engineering.enabled:
        return df.drop(columns=["Name"], errors="ignore")
    df = df.copy()
    title = df["Name"].str.extract(r",\s*([^.]*)\.", expand=False)
    df["Title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    return df.drop(columns=["Name"])
