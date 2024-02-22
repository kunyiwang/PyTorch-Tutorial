from torch.utils.data import Dataset
from PIL import Image
import os
from matplotlib import pyplot as plt

class Dataset(Dataset):

    def __init__(self, root_dir, label):
        self.root_dir = root_dir
        self.label = label
        self.path = os.path.join(self.root_dir, self.label)
        self.img_name_list = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_name_list[idx]
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path)

        return img
    
    def __len__(self):
        return len(self.img_name_list)
    
if __name__ == '__main__':
    root_dir = 'data/train'
    label = 'ants'
    dataset = Dataset(root_dir, label)

    # __len__
    print(f"Total images in the dataset: {len(dataset)}")

    # __getitem__
    img = dataset[1]
    img.show()