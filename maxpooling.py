import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

seed = 1234
torch.manual_seed(seed)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False))

    def forward(self, x):
        x = self.maxpool(x)
        return x

def maxpooling(feature_map, size=2, stride=1):
    # Preparing the output of the pooling operation.
    output_h = int(((feature_map.shape[2]-size) / stride) + 1)
    output_w = int(((feature_map.shape[3]-size) / stride) + 1)
    output_d = feature_map.shape[1]
    pool_out = np.zeros((output_d, output_h, output_w))
    # print(np.shape(pool_out))

    for map_num in range(output_d):
        r2 = 0
        for r in np.arange(0, feature_map.shape[2] - (size-1), stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[3] - (size-1), stride):
                # print(torch.max(feature_map[0][map_num, r:r + size, c:c + size]))
                # quit()
                pool_out[map_num, r2, c2] = torch.max(feature_map[0][map_num, r:r + size, c:c + size])
                c2 = c2 + 1
            r2 = r2 + 1
    pool_out = torch.tensor([pool_out])
    # print("size of maxpool output: ", pool_out.shape)
    return pool_out


if __name__ == '__main__':
    input = torch.randn(1, 3, 32, 32)
    dummy_input = torch.tensor([[[[1, 5, 8, 1], [6, 4, 6, 7], [1, 2, 4, 5], [5, 3, 3, 9]], [[2, 3, 0, 1], [4, 6, 8, 4], [1, 3, 7, 0], [5, 9, 2, 5]], [[1, 0, 4, 2], [7, 3, 8, 6], [1, 4, 0, 2], [3, 5, 6, 9]]]])
    # print(dummy_input.size())
    # print(input.size())

    # print(input)
    my_out = maxpooling(input)
    model = NeuralNet()
    torch_out = model(input)

    print("TORCH_OUT: ", torch_out)
    print("~"*100)
    print("MY_OUT: ", my_out)
    print(torch_out.shape)
    print(my_out.shape)
    print(np.array_equal(torch_out, my_out))
    print(torch.eq(torch_out, my_out))