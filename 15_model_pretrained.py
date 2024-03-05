import torchvision
import torch.nn as nn

if __name__ == '__main__':
    vgg16_false = torchvision.models.vgg16(pretrained=False) # Without pre-trained weight
    vgg16_true = torchvision.models.vgg16(pretrained=True) # With pre-trained weight

    vgg16_false.classifier[6] = nn.Linear(4096, 10) # Modify existing layer
    vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10)) # Add new layer

    print(vgg16_false)
    print(vgg16_true)