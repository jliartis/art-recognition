import torch.nn as nn
from torchvision.models import resnet152


class ResnetLarge(nn.Module):

    def __init__(self, num_classes):
        super(ResnetLarge, self).__init__()

        # ResnetLarge
        original_model = resnet152(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
