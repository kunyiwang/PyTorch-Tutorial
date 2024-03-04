import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2), # 3@32*32 -> 32@32*32
            nn.MaxPool2d(kernel_size=2), # 32@32*32 -> 32@16*16
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2), # 32@16*16 -> 32@16*16
            nn.MaxPool2d(kernel_size=2), # 32@16*16 -> 32@8*8
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2), # 32@8*8 -> 64@8*8
            nn.MaxPool2d(kernel_size=2), # 64@8*8 -> 64@4*4
            nn.Flatten(), # 1024
            nn.Linear(in_features=1024, out_features=64), # 1024 -> 64
            nn.ReLU(), # 64 -> 64
            nn.Linear(in_features=64, out_features=10) # # 64 -> 10
        )

    def forward(self, x):
        return self.model1(x)
    
if __name__ == '__main__':
    model = Model()
    input = torch.randn(64, 3, 32, 32)
    output = model(input)

    print(output.shape)

    writer = SummaryWriter('./logs')
    writer.add_graph(model, input)
    writer.close()