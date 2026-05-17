import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y


def get_resnet18(num_classes=102, pretrained=True):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet34(num_classes=102, pretrained=True):
    if pretrained:
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet18_se(num_classes=102, pretrained=True):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    for name, module in model.named_modules():
        if name in ["layer1", "layer2", "layer3", "layer4"]:
            for i, block in enumerate(module):
                se = SEBlock(block.conv2.out_channels)
                block.add_module("se", se)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet18_cbam(num_classes=102, pretrained=True):
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    for name, module in model.named_modules():
        if name in ["layer1", "layer2", "layer3", "layer4"]:
            for i, block in enumerate(module):
                cbam = CBAM(block.conv2.out_channels)
                block.add_module("cbam", cbam)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_vit_tiny(num_classes=102, pretrained=True):
    import timm
    if pretrained:
        model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    else:
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def get_swin_tiny(num_classes=102, pretrained=True):
    import timm
    if pretrained:
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
    else:
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model