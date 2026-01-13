def postprocess(pred):
    """
    Convert model output to human-readable response.
    """
    probability = float(pred[0][0])
    return {
        "probability": probability,
        "prediction": int(probability >= 0.5)
    }
