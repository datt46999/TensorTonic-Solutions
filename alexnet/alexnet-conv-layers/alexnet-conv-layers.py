import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation)."""
    batch, height, width, c = image.shape
    k = 11
    stride = 4
    filter = 96
    out_h = (height +4-k)//stride +1
    out_w = (width +4-k)//stride +1
    return np.zeros((batch, out_h, out_w, filter))