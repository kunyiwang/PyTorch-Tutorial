from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

if __name__ == '__main__':
    img_path = 'data/train/ants/0013035.jpg'
    img = Image.open(img_path)
    # img.show()
    writer = SummaryWriter('logs')

    trans = transforms.ToTensor()
    img_tensor = trans(img)

    writer.add_image('Tensor Image', img_tensor)
    writer.close()
