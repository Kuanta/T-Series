import numpy as np

def nth_smoothing(data: np.ndarray, n: int, factor: float):
    '''
    Implements nth order smoothing to a time series
    :param data:
    :param n: Order of smoothing
    :param factor: Smoothing factor
    :return:
    '''
    data_size = data.size
    smoothed = np.zeros(shape=data.shape)
    last_smooth = np.ones(shape=n)*data[0]  #last_smooth[0] is the first order smoothed value
    for i in range(data_size):
        curr_smooth = np.ones(shape=n)
        curr_smooth[0] = factor*data[i]+(1-factor)*last_smooth[0]#First order smoothing
        for j in range(1,n):
            curr_smooth[j] = factor*curr_smooth[j-1] + (1-factor)*last_smooth[j]
        smoothed[i] = curr_smooth[-1]  # Last value of the curr_smooth is the smoothed value of highest order
    return smoothed
