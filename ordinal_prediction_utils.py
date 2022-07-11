def prediction2label(pred):
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1
