import torch
import torch.nn as nn
from torchvision.models import resnet152, regnet_y_800mf


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


class RegNet(nn.Module):

    def __init__(self, num_classes, model, frozen_layers):
        super(RegNet, self).__init__()
        original_model = model(pretrained=True)
        self.frozen_layers = frozen_layers
        self.features = nn.Sequential(original_model.stem, *list(original_model.trunk_output.children()))
        for name, child in list(self.features.named_children())[:frozen_layers]:
            for param in child.parameters():
                param.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def finetune(self, n_layers, optimizer):
        for name, child in list((self.features[1]).named_children())[n_layers:self.frozen_layers]:
            for param in child.parameters():
                param.requires_grad = True
            optimizer.add_param_group({'params': child.parameters()})
