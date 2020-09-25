from datagen.datagen import gen_sinusoidal
from forecast.least_squares import least_squares
import matplotlib.pyplot as plt
import numpy as np
from process.smoothing import nth_smoothing

train = np.linspace(0,500-1,500)*0.1
test = np.linspace(500,1000-1,500)*0.1
y = gen_sinusoidal(train, freq=0.9, phase=0, dc_gain=0, noise_variance=0.6)
smooth_y = nth_smoothing(y, 2, 0.3)
lin_y = 2.5*train+0.3
weights, bias = least_squares(train, lin_y, iteration=500)
print(weights)
print(bias)
pred = test*weights+bias
print(pred)
#plt.plot(train,lin_y)
#plt.plot(test,pred)
plt.plot(y)
plt.plot(smooth_y)
plt.show()