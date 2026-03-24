import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    # wx whh
    batch, len_seq, input_size = X.shape
    h= h_0
    h_list = []
    for t in range(len_seq):
        xth = X[:, t, :]  
        
        h_1= np.dot(h_0, W_hh)    
        h_2 = np.dot(xth, W_xh.T) 
        h = np.tanh(h_1 + h_2+ b_h)
        h_list.append(h)

    result = np.stack(h_list, axis =1)
    return result, h