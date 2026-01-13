import numpy as np

def preprocess(features):
    """
    Basic preprocessing hook.
    Assumes features are already numeric and ordered.
    """
    return np.array(features, dtype=np.float32)
