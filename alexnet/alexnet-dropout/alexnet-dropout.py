import numpy as np

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    if training:
        return np.random.binomial(1, .5, len(x)) 
    return x