import torch
import torch.nn as nn
import torchvision.models as models

class mix_model(nn.Module):
    def __init__(self):
        super(mix_model, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 512)
        self.linear1 = nn.Linear(512, 2)
        self.linear2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet(x)
        reg = self.linear1(x)
        cls = self.linear2(x)
        cls = torch.sigmoid(cls)

        return reg, cls