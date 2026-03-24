import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    hpre = np.dot(h_prev, W_hh.T)
    hx = np.dot(x_t, W_xh.T) 
    ht = np.tanh(hpre + hx + b_h)
    return ht