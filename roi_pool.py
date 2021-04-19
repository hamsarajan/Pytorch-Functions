import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math

seed = 1234
torch.manual_seed(seed)

def my_roi_pool(input, boxes, output_size):
    boxes = boxes[0]
    roi_start_h = int(boxes[1])
    roi_start_w = int(boxes[2])
    roi_end_h = int(boxes[3])
    roi_end_w = int(boxes[4])

    roi_width = roi_end_w - roi_start_w + 1
    roi_height = roi_end_h - roi_start_h + 1

    size_w = roi_width / output_size[0]
    size_h = roi_height / output_size[1]
    output = torch.zeros((input.shape[0], input.shape[1], output_size[0], output_size[1]))

    for channel in range(input.shape[1]):
        for j in range(output_size[0]):
            for i in range(output_size[1]):
                output[0,channel,j,i] = (input[0, channel, int(size_h*(j)):math.ceil(size_h*(j+1)), int(size_w*(i)): math.ceil(size_w*(i+1))]).max()
    print(output.shape)
    return output

if __name__ == '__main__':
    input = torch.randn(1, 3, 32, 32)
    dummy_input = torch.tensor([[[[1, 5, 8, 1], [6, 4, 6, 7], [1, 2, 4, 5], [5, 3, 3, 9]],
                                 [[2, 3, 0, 1], [4, 6, 8, 4], [1, 3, 7, 0], [5, 9, 2, 5]],
                                 [[1, 0, 4, 2], [7, 3, 8, 6], [1, 4, 0, 2], [3, 5, 6, 9]]]])

    rois = torch.tensor([
        [0.0, 0.0, 0.0, 2.0, 2.0],
    ])
    output_size = (2,2)
    # dummy_input = dummy_input.type(torch.FloatTensor)
    # small_input = small_input.type(torch.FloatTensor)
    # print(dummy_input)
    torch_out = torchvision.ops.roi_pool(input=input, boxes=rois, output_size=output_size, spatial_scale=1)
    my_out = my_roi_pool(input, rois, output_size)

    print("TORCH_OUT: ", torch_out)
    print("~" * 100)
    print("MY_OUT: ", my_out)
    print(torch_out.shape)
    print(my_out.shape)
    print(np.array_equal(torch_out, my_out))
    print(torch.eq(torch_out, my_out))