import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math

seed = 1234
torch.manual_seed(seed)
np.random.seed(1234)

def softmax(input):
    exps = torch.exp(input)
    sum_exps = torch.sum(exps, 1, keepdim=True)
    return(exps/sum_exps)

def flatten(input):
    h = input.shape[2]
    w = input.shape[3]
    output = input[0].view(-1, h*w)
    return output

def crossEntropy(input, target):
    loss = 0
    for i in range(input.shape[0]):
        loss -= torch.log(softmax(input))[i][target[i]]
    return loss/target.shape[0]

if __name__ == '__main__':
    # example()
    # quit()
    input = torch.randn(1, 3, 32, 32)
    target = torch.tensor([1, 0, 0])

    dummy_input = torch.tensor([[[[1, 5, 8, 1], [6, 4, 6, 7], [1, 2, 4, 5], [5, 3, 3, 9]],
                                 [[2, 3, 0, 1], [4, 6, 8, 4], [1, 3, 7, 0], [5, 9, 2, 5]],
                                 [[1, 0, 4, 2], [7, 3, 8, 6], [1, 4, 0, 2], [3, 5, 6, 9]]]])
    input = flatten(input)
    input = input.type(torch.FloatTensor)
    print(input.shape)
    print(target.shape)
    # quit()
    torch_out = F.cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    my_out = crossEntropy(input, target)


    print("TORCH_OUT: ", torch_out)
    print("~" * 100)
    print("MY_OUT: ", my_out)
    print(np.array_equal(torch_out, my_out))
    print(torch.eq(torch_out, my_out))