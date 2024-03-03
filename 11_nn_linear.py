import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(196608, 10) # 196608 = 64*3*32*32

    def forward(self, input):
        return self.linear1(input)

if __name__ == '__main__':
    dataset_transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transform, download=False)
    dataLoader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

    model = Model()

    step = 0
    for batch in dataLoader:
        imgs, targets = batch
        imgs = torch.flatten(imgs) # Or: imgs = torch.reshape(imgs, (1, -1))
        print(imgs.shape)
        output = model(imgs)
        print(output.shape)
        step = step + 1