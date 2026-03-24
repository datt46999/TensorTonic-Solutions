import numpy as np
import math
def sigmoid(x):
    x = np.asarray(x, dtype = float)
    return (1/(1+ np.exp(-x)))