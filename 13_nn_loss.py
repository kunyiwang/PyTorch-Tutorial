import torch
import torch.nn as nn

if __name__ == '__main__':
    input = torch.tensor([1, 2, 3], dtype=torch.float32)
    target = torch.tensor([1, 3, 5], dtype=torch.float32)

    input = torch.reshape(input, [1, 1, 1, 3])
    target = torch.reshape(target, [1, 1, 1, 3])

    loss_L1 = nn.L1Loss()
    loss_MSE = nn.MSELoss()

    result_L1 = loss_L1(input, target)
    result_MSE = loss_MSE(input, target)

    print(result_L1)
    print(result_MSE)

    # CrossEntropy Loss
    loss_cross = nn.CrossEntropyLoss()
    x = torch.tensor([0.1, 0.2, 0.3])
    x = torch.reshape(x, [1, 3])
    y = torch.tensor([1])
    result_cross = loss_cross(x, y)
    print(result_cross)