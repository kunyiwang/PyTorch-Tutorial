from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

if __name__ == '__main__':
    writer = SummaryWriter('logs')

    for i in range(100):
        writer.add_scalar('y=2x', 2*i, i)

    img_path_val = 'data/val/ants/800px-Meat_eater_ant_qeen_excavating_hole.jpg'
    img_path_train = 'data/train/ants/0013035.jpg'

    img1 = np.array(Image.open(img_path_train))
    img2 = np.array(Image.open(img_path_val))

    writer.add_image('image', img1, 1, dataformats='HWC')
    writer.add_image('image', img2, 2, dataformats='HWC')

    writer.close()

# Following is command to visualize the log file:
# tensorboard --logdir=logs