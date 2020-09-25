import numpy as np

def gen_sinusoidal(input, freq=1, phase=0, dc_gain=0, noise_variance=1):
    noise = np.random.normal(0, noise_variance, size=np.size(input))
    y = np.sin(freq*input+phase)+dc_gain
    y = y + noise
    return y

