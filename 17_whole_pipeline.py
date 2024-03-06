import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import os

class MyDataset(Dataset):

    def __init__(self, root_dir, label, transform):
        self.root_dir = root_dir
        self.label = label
        self.path = os.path.join(self.root_dir, self.label)
        self.img_name_list = os.listdir(self.path)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.img_name_list[idx]
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path)
        img = self.transform(img)
        if self.label == 'ants':
            target = 0
        else:
            target = 1

        return img, target
    
    def __len__(self):
        return len(self.img_name_list)
    
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
            nn.Linear(in_features=64, out_features=16), # 64 -> 16
            nn.ReLU(), # 16 -> 16
            nn.Linear(in_features=16, out_features=2) # 16 -> 2
        )

    def forward(self, x):
        return self.model1(x)
    
if __name__ == '__main__':

    # Set Transformation
    dataset_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor()
    ])

    # Set Dataset
    root_dir_train = './data/train'
    root_dir_test = './data/val'
    ants_label = 'ants'
    bees_label = 'bees'
    ants_dataset = MyDataset(root_dir_train, ants_label, dataset_transform)
    bees_dataset = MyDataset(root_dir_train, bees_label, dataset_transform)
    train_dataset = ants_dataset + bees_dataset
    ants_dataset = MyDataset(root_dir_test, ants_label, dataset_transform)
    bees_dataset = MyDataset(root_dir_test, bees_label, dataset_transform)
    test_dataset = ants_dataset + bees_dataset

    print(f"Total images in the training dataset: {len(train_dataset)}")
    print(f"Total images in the training dataset: {len(test_dataset)}")

    # Set DataLoader
    train_dataLoader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    test_dataLoader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)

    # Set Model
    model = Model()

    # Set Loss Function
    loss_fn = nn.CrossEntropyLoss()

    # Set Optimizer
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Set Parameters
    total_train_steps = 0
    total_test_steps = 0
    epoch = 10

    # Set Tensorboard
    writer = SummaryWriter('./logs')

    # Set Training Loop
    model.train()
    for i in range(epoch):
        print('-----Epoch {} Starts:-----'.format(i+1))
        for batch in train_dataLoader:
            imgs, targets = batch
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad() # Initilize the gradient
            loss.backward() # Calculate the gradient
            optimizer.step() # Gradient descent

            total_train_steps = total_train_steps + 1
            if total_train_steps % 1 == 0:
                print('Training batch counts:{}, Loss:{}'.format(total_train_steps, loss.item()))
                writer.add_scalar('Training Loss', loss.item(), total_train_steps)

        # Set test for each epoch
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch in test_dataLoader:
                imgs, targets = batch
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item() # loss' type is tensor, loss.item() is float
        print('Test Set Loss:{}'.format(total_test_loss))
        total_test_steps = total_test_steps + 1
        writer.add_scalar('Test Loss:', total_test_loss, total_test_steps)

        # Save Model
        torch.save(model.state_dict(),'./models/kunyi_{}'.format(i))
        print('Model Saved!')
    
    writer.close()

    

