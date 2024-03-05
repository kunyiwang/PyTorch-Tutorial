import torch
import torchvision
import torch.nn as nn

if __name__ == '__main__':
    model = torchvision.models.vgg16(pretrained=True)

    # Method 1, both model structure&weight are saved(Not recommended)
    torch.save(model, './models/vgg16_1.pth')
    vgg_1 = torch.load('./models/vgg16_1.pth')

    # Method 2, only model weight is saved(Recommended since memory efficient)
    torch.save(model.state_dict(), './models/vgg16_2.pth')
    vgg_2 = torchvision.models.vgg16(pretrained=False)
    state_dict = torch.load('./models/vgg16_2.pth')
    vgg_2.load_state_dict(state_dict)

    # Something Specific -- if u comment out the class Model(), the load will be failed, because compiler will be unsure about what is model
    # In conclusion, method 1 is bad!!!
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3)

        def forward(self, x):
            return self.conv1(x)
    
    model = Model()
    torch.save(model, './models/model_1.pth')
    model_1 = torch.load('./models/model_1.pth')