import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    dataset_transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transform, download=False)

    # img, target = test_set[0]
    # print(img.shape)
    # print(target)

    test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

    writer = SummaryWriter('logs')
    for epoch in range(2):
        step = 0
        for data in test_loader:
            imgs, targets = data
            # print(imgs.shape)
            # print(targets)
            writer.add_images('Epoch: {}'.format(epoch), imgs, step) # Note: add_images instead of add_image
            step = step + 1

    writer.close()
    