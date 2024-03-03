import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.relu1 = nn.ReLU()
        self.sigmod1 = nn.Sigmoid()

    def forward(self, input):
        return self.sigmod1(input)

if __name__ == '__main__':
    dataset_transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transform, download=False)
    dataLoader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

    input = torch.tensor([[1, 2, -3, 4, -5],
                          [-1, 2, 3, -4, 5],
                          [1, -2, 3, 4, -5],
                          [1, 2, -3, -4, 5],
                          [-1, 2, 3, 4, -5]], dtype=torch.float32)
    input = torch.reshape(input, [1, 1, 5, 5]) # for second parameter, both [] and () are fine

    model = Model()
    output = model(input)
    print(output)

    writer = SummaryWriter('./logs')
    step = 0
    for batch in dataLoader:
        imgs, targets = batch
        # print(imgs.shape)
        output = model(imgs)
        # print(output.shape)
        writer.add_images('input', imgs, step)
        writer.add_images('output', output, step)
        step = step + 1

    writer.close()