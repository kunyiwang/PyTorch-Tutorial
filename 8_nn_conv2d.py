import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, input):
        return self.conv1(input)
    
if __name__ == '__main__':
    dataset_transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transform, download=False)
    dataLoader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

    model = Model()
    writer = SummaryWriter('./logs')

    step = 0
    for batch in dataLoader:
        imgs, targets = batch
        # print(imgs.shape)
        output = model(imgs)
        # print(output.shape)
        output = torch.reshape(output, (-1, 3, 30, 30)) # tensorboard can only display images with 3 dimensions(RGB)
        writer.add_images('input', imgs, step)
        writer.add_images('output', output, step)
        step = step + 1

    writer.close()