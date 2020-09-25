import numpy as np

def least_squares(regressors, outputs, step_size = 0.001, iteration = 100):
    '''
    Linear least squares regression
    :param regressors:
    :param outputs:
    :return: Param vector
    '''
    data_size = np.size(regressors)
    if len(np.shape(regressors)) == 1:
        regressors = regressors.reshape((-1, 1))
    batch_size = np.size(outputs)
    if(len(np.shape(regressors)) == 1):
        dim_size = 1
    else:
        dim_size = np.shape(regressors)[1]
    weights = np.random.normal(size=dim_size)
    bias = np.ones(shape=1)
    for i in range(iteration):
        _grad = outputs.reshape(-1,1)-bias-weights*regressors
        grad_w = -2 * np.sum(np.transpose(_grad*regressors), axis=1)
        grad_b = -2 * np.sum(np.transpose(_grad), axis=1)
        weights = weights - step_size*grad_w
        bias = bias - step_size*grad_b
    return weights, bias

def weighted_least_squares():
    pass