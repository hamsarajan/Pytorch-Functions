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
        self.convTran2d = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0, groups=1, bias=True, dilation=dilation, padding_mode='zeros')
        )

    def forward(self, x):
        x = self.convTran2d(x)
        return x

def convtranspose2d(input, in_channels, out_channels, size, filter, bias):

    channels = in_channels
    rows = input.shape[2]
    col = input.shape[3]
    h = filter.shape[2]
    w = filter.shape[3]
    conv_out = torch.zeros((1, out_channels, input.shape[2] + h -1, input.shape[3] + w -1))

    result = []
    for out_channel in range(out_channels):
        for channel in range(channels):
            for r in range(rows):
                for c in range(col):
                    conv_out[0, out_channel, r:r + size, c:c + size] += input[0, channel, r, c] * filter[channel, out_channel]
        conv_out[0, out_channel] = conv_out[0, out_channel] + bias[out_channel]
    # print(conv_out.shape)
    return conv_out


if __name__ == '__main__':
    input = torch.randn(1, 3, 32, 32)
    dummy_input = torch.tensor([[[[1, 5, 8, 1], [6, 4, 6, 7], [1, 2, 4, 5], [5, 3, 3, 9]],
                                 [[2, 3, 0, 1], [4, 6, 8, 4], [1, 3, 7, 0], [5, 9, 2, 5]],
                                 [[1, 0, 4, 2], [7, 3, 8, 6], [1, 4, 0, 2], [3, 5, 6, 9]]]])
    in_channels = 3
    out_channels = 4
    kernel_size = 3
    stride = 1
    dilation = 1
    model = NeuralNet(in_channels, out_channels, kernel_size, stride, dilation)
    dummy_input = dummy_input.type(torch.FloatTensor)

    torch_out = model(input)
    parameters = []
    for param in model.parameters():
        parameters.append(param)
    filter = parameters[0]
    bias = np.array((parameters[1]).detach())
    my_out = convtranspose2d(input, in_channels, out_channels, kernel_size, filter, bias)

    #rounding off the answer to 4 decimal places
    n_digits = 4
    my_out = torch.round(my_out * 10 ** n_digits) / (10 ** n_digits)
    torch_out = torch.round(torch_out * 10 ** n_digits) / (10 ** n_digits)

    print(torch_out.dtype, my_out.dtype)
    print("TORCH_OUT: ", torch_out)
    print("~" * 100)
    print("MY_OUT: ", my_out)
    print(torch_out.shape)
    print(my_out.shape)
    print(torch.equal(torch_out, my_out))