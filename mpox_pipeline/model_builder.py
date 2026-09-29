import torch
import torch.nn as nn
import timm

BACKBONES = {
    "ResNet50V2": "resnet50d",
    "DenseNet121": "densenet121",
    "DenseNet201": "densenet201",
    "MobileNetV2": "mobilenetv2_100",
    "EfficientNetB0": "efficientnet_b0",
    "EfficientNetB3": "efficientnet_b3",
}

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.fc(x).view(b, c, 1, 1)
        return x * w

class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.se = SEBlock(channels, reduction)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.se(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.spatial_conv(spatial)
        return x * spatial

class CustomClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes, attention_type="none", attention_depth="single"):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        in_features = self.backbone.num_features

        attention_modules = []
        depth_count = 1 if attention_depth == "single" else 2
        for _ in range(depth_count):
            if attention_type == "se":
                attention_modules.append(SEBlock(in_features))
            elif attention_type == "eca":
                attention_modules.append(ECABlock(in_features))
            elif attention_type == "cbam":
                attention_modules.append(CBAMBlock(in_features))

        self.attention = nn.Sequential(*attention_modules) if attention_modules else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.attention(x)
        if len(x.shape) == 4:
            x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_model(backbone_key, num_classes, attention_type="none", attention_depth="single"):
    timm_name = BACKBONES.get(backbone_key, "resnet50d")
    model = CustomClassifier(timm_name, num_classes, attention_type, attention_depth)
    img_size = 224
    return model, img_size