from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

if __name__ == '__main__':
    img_path = 'data/train/ants/0013035.jpg'
    img = Image.open(img_path)
    # img.show()
    writer = SummaryWriter('logs')

    # ToTensor PIL -> Tensor
    trans_toTensor = transforms.ToTensor()
    img_tensor = trans_toTensor(img)
    writer.add_image('Tensor Image', img_tensor)

    # Normalize PIL -> Tensor -> Normalize
    trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img_norm = trans_norm(img_tensor)
    writer.add_image('Normalized Tensor Image', img_norm)

    # Resize PIL -> Tensor -> Resize
    # print(img_tensor.size())
    trans_resize = transforms.Resize((512, 512))
    img_resized = trans_resize(img_tensor)
    writer.add_image('Resized Tensor Image', img_resized)
    # print(img_resized.size())

    # Compose PIL -> Tensor -> Normalize -> Resize
    trans_compose = transforms.Compose([trans_toTensor, trans_norm, trans_resize])
    img_composed = trans_compose(img)
    writer.add_image('Composed Instructions to Image', img_composed)

    # RandomCrop PIL -> Tensor -> Crop
    trans_crop = transforms.RandomCrop((128,256))
    trans_compose = transforms.Compose([trans_toTensor, trans_crop])
    for i in range(10):
        img_crop = trans_compose(img)
        writer.add_image('Cropped Image', img_crop, i)

    writer.close()
