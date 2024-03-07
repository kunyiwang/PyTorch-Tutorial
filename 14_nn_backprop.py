import torch
import time
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
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
    device = torch.device('cpu')
    dataset_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=dataset_transform, download=False)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transform, download=False)

    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

    model = Model()
    model.to(device)
    loss = nn.CrossEntropyLoss()
    loss.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(10):
        running_loss = 0.0
        start_time = time.time()
        for batch in train_loader:
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = model(imgs)
            result_loss = loss(output, targets)
            optimizer.zero_grad() # Initilize the gradient
            result_loss.backward() # Calculate the gradient
            optimizer.step() # Using the gradient in previous step to achieve gradient descent(which is applied to model.parameters())
            running_loss = running_loss + result_loss
        print('Epoch Training Time:{}'.format(time.time() - start_time))

        print(running_loss)
