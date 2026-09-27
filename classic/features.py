import pandas as pd


def feature_engineering(df: pd.DataFrame, config) -> pd.DataFrame:
    
    if not config.feature_engineering.enabled:
        return df
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")
    common_titles = ["Mr", "Mrs", "Miss", "Master"]
    df["Title"] = df["Title"].where(df["Title"].isin(common_titles),"Rare")
    df = df.drop(columns=["Name"])
    return df
