import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        # 特征提取部分
        self.features = features
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 全连接层，输入512*7*7，输出4096
            nn.ReLU(True),  # 激活函数ReLU
            nn.Dropout(),  # Dropout层，防止过拟合
            nn.Linear(4096, 4096),  # 第二个全连接层
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),  # 最后一层输出类别数
        )

    def forward(self, x):
        # 前向传播
        x = self.features(x)  # 提取特征
        x = torch.flatten(x, 1)  # 展平特征图
        x = self.classifier(x)  # 分类
        return x

def make_layers(cfg, batch_norm=False):
    # 根据配置创建卷积层
    layers = []
    in_channels = 3  # 输入通道数，RGB图像为3
    for v in cfg:
        if v == 'M':
            # 最大池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 卷积层
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # 如果使用批归一化
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v  # 更新输入通道数
    return nn.Sequential(*layers)

# VGG16的配置
cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

def vgg16(num_classes=1000, batch_norm=False):
    # 创建VGG16模型
    return VGG(make_layers(cfg['VGG16'], batch_norm=batch_norm), num_classes=num_classes)

if __name__ == "__main__":
    # 测试模型，假设有10个类别
    model = vgg16(num_classes=10)
    print(model)