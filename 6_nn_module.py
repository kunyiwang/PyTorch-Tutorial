import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input):
        return input*2
    

if __name__ == '__main__':
    model = Model()
    input = torch.tensor([[1, 2, -3, 4, -5],
                          [-1, 2, 3, -4, 5],
                          [1, -2, 3, 4, -5],
                          [1, 2, -3, -4, 5],
                          [-1, 2, 3, 4, -5],])
    output = model(input)
    print(output)