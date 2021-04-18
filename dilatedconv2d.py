import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math

seed = 1234
torch.manual_seed(seed)
np.random.seed(1234)

class NeuralNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(NeuralNet, self).__init__()
        self.dilatedconv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=1, bias=True, padding_mode='zeros')
        )

    def forward(self, x):
        x = self.dilatedconv2d(x)
        return x

def prepareFilter(filter, out_channels, in_channels, size, dilation):
    dilated_size = (dilation*size) - 1
    dilated_filter = np.zeros((out_channels, in_channels, dilated_size, dilated_size))
    for filtr in range(out_channels):
        for channel in range(in_channels):
            for r in range(size):
                for c in range(size):
                    dilated_filter[filtr, channel, r*dilation, c*dilation] = filter[filtr, channel, r, c]
    return dilated_size, dilated_filter

def dilatedconv2d(feature_map, in_channels, out_channels, size, stride, filter, bias):
    output_h = int(((feature_map.shape[2] - size) / stride) + 1)
    output_w = int(((feature_map.shape[3] - size) / stride) + 1)
    output_d = out_channels
    conv_out = np.zeros((output_d, output_h, output_w))

    for filtr in range(out_channels):
        r2 = 0
        for r in np.arange(0, feature_map.shape[2] - (size-1), stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[3] - (size-1), stride):
                sum_channel = 0
                for channel in range(in_channels):
                    sum_channel += torch.sum((filter[filtr, channel] * feature_map[0][channel, r: r + size, c: c + size]))
                conv_out[filtr, r2, c2] = sum_channel
                c2 += 1
            r2 += 1
        conv_out[filtr] = conv_out[filtr] + bias[filtr]
    conv_out = torch.tensor([conv_out])
    # print("size of conv output: ", conv_out.shape)
    return conv_out


if __name__ == '__main__':
    input = torch.randn(1, 3, 32, 32)
    dummy_input = torch.tensor([[[[1, 5, 8, 1], [6, 4, 6, 7], [1, 2, 4, 5], [5, 3, 3, 9]],
                                 [[2, 3, 0, 1], [4, 6, 8, 4], [1, 3, 7, 0], [5, 9, 2, 5]],
                                 [[1, 0, 4, 2], [7, 3, 8, 6], [1, 4, 0, 2], [3, 5, 6, 9]]]])
    in_channels = 3
    out_channels = 6
    kernel_size = 5
    stride = 2
    dilation = 2
    model = NeuralNet(in_channels, out_channels, kernel_size, stride, dilation)
    dummy_input = dummy_input.type(torch.FloatTensor)

    torch_out = model(input)
    # print(torch_out.shape)
    parameters = []
    for param in model.parameters():
        parameters.append(param)
    filter = parameters[0]
    bias = np.array((parameters[1]).detach())

    dilated_size, dilated_filter = prepareFilter(filter, out_channels, in_channels, kernel_size, dilation)
    dilated_filter = torch.from_numpy(dilated_filter)
    my_out = dilatedconv2d(input, in_channels, out_channels, dilated_size, stride, dilated_filter, bias)

    #rounding off the answer to 4 decimal places
    torch_out = torch_out.type(torch.DoubleTensor)
    n_digits = 3
    my_out = torch.round(my_out * 10 ** n_digits) / (10 ** n_digits)
    torch_out = torch.round(torch_out * 10 ** n_digits) / (10 ** n_digits)

    print("TORCH_OUT: ", torch_out)
    print("~" * 100)
    print("MY_OUT: ", my_out)
    print(torch_out.shape)
    print(my_out.shape)
    print(torch.equal(torch_out, my_out))