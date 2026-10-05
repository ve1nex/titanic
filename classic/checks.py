def check_data_leakage(X, y, config):
    """Reject a target column accidentally included among features."""
    if str(config.data.target) in X.columns:
        raise ValueError("Target column leaked into features")
