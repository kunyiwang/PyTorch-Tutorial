from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataset(Dataset):

    def __init__(self, root_dir, label):
        self.root_dir = root_dir
        self.label = label
        self.path = os.path.join(self.root_dir, self.label)
        self.img_name_list = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_name_list[idx]
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path)

        return img, self.label
    
    def __len__(self):
        return len(self.img_name_list)
    
if __name__ == '__main__':
    root_dir = 'data/train'
    ants_label = 'ants'
    bees_label = 'bees'
    ants_dataset = MyDataset(root_dir, ants_label)
    bees_dataset = MyDataset(root_dir, bees_label)
    train_dataset = ants_dataset + bees_dataset

    # __len__
    print(f"Total images in the dataset: {len(train_dataset)}")

    # __getitem__
    img, label = train_dataset[1]
    print(label)
    img.show()