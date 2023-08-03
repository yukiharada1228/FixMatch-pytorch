import timm
import torch
import torch.nn as nn


def resnet(depth, num_classes, pretrained=False):
    return timm.create_model(
        f"resnet{depth}", pretrained=pretrained, num_classes=num_classes
    )


def resnet18(num_classes, pretrained=False):
    return resnet(depth=18, num_classes=num_classes, pretrained=pretrained)


def resnet34(num_classes, pretrained=False):
    return resnet(depth=34, num_classes=num_classes, pretrained=pretrained)


def resnet50(num_classes, pretrained=False):
    return resnet(depth=50, num_classes=num_classes, pretrained=pretrained)


def cifar_resnet34(num_classes, pretrained=False):
    model = resnet34(num_classes=num_classes, pretrained=pretrained)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)
    model.maxpool = nn.Identity()
    init_weights(model=model)
    return model


@torch.jit.ignore
def init_weights(model, zero_init_last=True):
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    if zero_init_last:
        for m in model.modules():
            if hasattr(m, "zero_init_last"):
                m.zero_init_last()
