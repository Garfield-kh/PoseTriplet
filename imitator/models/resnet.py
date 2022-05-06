from torch import nn
from torchvision import models
from utils.torch import *


class ResNet(nn.Module):

    def __init__(self, out_dim, fix_params=False):
        super().__init__()
        self.out_dim = out_dim
        self.resnet = models.resnet18(pretrained=True)
        if fix_params:
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_dim)

    def forward(self, x):
        return self.resnet(x)


if __name__ == '__main__':
    import time
    net = ResNet(128)
    t0 = time.time()
    input = ones(64, 3, 224, 224)
    out = net(input)
    print(time.time() - t0)
    print(out.shape)
