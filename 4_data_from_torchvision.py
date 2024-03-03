import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

if __name__ == '__main__':
    dataset_transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=dataset_transform, download=False)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transform, download=False)

    # img, label_idx = test_set[0]
    # print(label_idx)
    # print(test_set.classes[label_idx])

    writer = SummaryWriter('logs')
    for i in range(10):
        img, _ = train_set[i]
        writer.add_image('train image', img, i)

    writer.close()