import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math

seed = 1234
torch.manual_seed(seed)
np.random.seed(1234)

def mean_var(input):
    input = np.array(input)
    filters = input.shape[0]
    channels = input.shape[1]
    running_mean = np.zeros(3)
    running_var = np.zeros(3)
    sum_mean = np.zeros((4,4))
    for channel in range(channels):
        for filtr in range(filters):
            # sum_mean += input[filtr, channel]
            if filtr == 0:
                attached = input[filtr, channel]
            else:
                attached = np.concatenate(input[filtr, channel])
        # print((np.mean(attached, axis=0)))
        running_mean[channel] = np.mean(attached, axis=0)
        running_var[channel] = np.std(attached, axis=0)

    running_mean = torch.from_numpy(running_mean)
    running_var = torch.from_numpy(running_var)
    return running_mean, running_var

def batch_norm(input, running_mean, running_var, eps, gamma, beta):
    C = input.shape[1]
    # X_hat = (X - mean) * 1.0 / nd.sqrt(variance + eps)
    X_hat = (input - running_mean.reshape((1, C, 1, 1))) * 1.0 / np.sqrt(running_var.reshape((1, C, 1, 1)) + eps)
    out = gamma * X_hat + beta
    return out

if __name__ == '__main__':
    input = torch.randn(2, 3, 32, 32)
    dummy_input = torch.tensor([[[[1, 5, 8, 1], [6, 4, 6, 7], [1, 2, 4, 5], [5, 3, 3, 9]],
                                 [[2, 3, 0, 1], [4, 6, 8, 4], [1, 3, 7, 0], [5, 9, 2, 5]],
                                 [[1, 0, 4, 2], [7, 3, 8, 6], [1, 4, 0, 2], [3, 5, 6, 9]]],
                                [[[2, 3, 4, 5], [6, 2, 7,3], [2, 8, 4, 6], [1, 0, 4, 7]],
                                [[8, 2, 5, 6], [9, 3, 7, 4], [1, 7, 9, 0], [5, 7, 6, 3]],
                                [[3, 4, 6, 5], [9, 5, 7, 3], [3, 7, 4, 5], [2, 7, 3, 6]]]])
    momentum = 0.1
    eps = 1e-05
    gamma = 1
    beta = 0
    running_mean, running_var = mean_var(input)
    running_mean = running_mean.type(torch.FloatTensor)
    running_var = running_var.type(torch.FloatTensor)
    my_out = batch_norm(input, running_mean, running_var, eps, gamma, beta)
    torch_out = F.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=momentum, eps=eps)

    #rounding off to the nearest 4 decimal places
    n_digits = 4
    my_out = torch.round(my_out * 10 ** n_digits) / (10 ** n_digits)
    torch_out = torch.round(torch_out * 10 ** n_digits) / (10 ** n_digits)

    print("TORCH_OUT: ", torch_out)
    print("~" * 100)
    print("MY_OUT: ", my_out)
    print(torch_out.shape)
    print(my_out.shape)
    print(torch.eq(torch_out, my_out))
    # print(torch.allclose(torch_out, my_out))
