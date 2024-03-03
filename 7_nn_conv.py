import torch
import torch.nn.functional as F

if __name__ == '__main__':
    input = torch.tensor([[1, 2, -3, 4, -5],
                          [-1, 2, 3, -4, 5],
                          [1, -2, 3, 4, -5],
                          [1, 2, -3, -4, 5],
                          [-1, 2, 3, 4, -5]], dtype=torch.float32)
    
    kernal = torch.tensor([[[1, 2, 1],
                           [0, 1, 0],
                           [2, 1, 0]],
                           [[1, 2, 1],
                           [0, 1, 0],
                           [2, 1, 0]],
                           [[1, 2, 1],
                           [0, 1, 0],
                           [2, 1, 0]]], dtype=torch.float32)
    
    # Reshaping because of the parameter reuqirement:
    #   input - (minibatch, in_channels, input_H, input_W)
    #       in_channels, for RGB images, it should be 3
    #   weight - (out_channels, in_channels/groups, kernal_H, kernal_W)
    #       groups in weight is usually set as 1, groups is a parameter in F.conv2d()
    input = torch.reshape(input, [1, 1, 5, 5])
    kernal = torch.reshape(kernal, [3, 1, 3, 3]) # weight

    output = F.conv2d(input, kernal, stride=1, padding=0)

    print(output.shape)
    print(output)