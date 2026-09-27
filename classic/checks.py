def check_data_leakage(X, y, config):
    if str(config.data.target) in X.columns:
        raise ValueError("Target column leaked into features")
