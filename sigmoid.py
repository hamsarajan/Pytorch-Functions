import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math

seed = 1234
torch.manual_seed(seed)
np.random.seed(1234)

if __name__ == '__main__':
    input = torch.randn(1, 3, 32, 32)
    dummy_input = torch.tensor([[[[1, 5, 8, 1], [6, 4, 6, 7], [1, 2, 4, 5], [5, 3, 3, 9]],
                                 [[2, 3, 0, 1], [4, 6, 8, 4], [1, 3, 7, 0], [5, 9, 2, 5]],
                                 [[1, 0, 4, 2], [7, 3, 8, 6], [1, 4, 0, 2], [3, 5, 6, 9]]]])
    my_out = 1.0/ (1.0 + np.exp(-input))
    torch_out = torch.sigmoid(input)

    print("TORCH_OUT: ", torch_out)
    print("~" * 100)
    print("MY_OUT: ", my_out)
    print(torch_out.shape)
    print(my_out.shape)
    print(np.array_equal(torch_out, my_out))
    print(torch.eq(torch_out, my_out))