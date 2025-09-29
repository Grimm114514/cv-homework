# ----------------------------------
# 导入所有需要的库
# ----------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import urllib.request
import gzip
import shutil

# 手动指定国内镜像源
resources = [
    ("https://mirrors.tuna.tsinghua.edu.cn/git/MNIST/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz"),
    ("https://mirrors.tuna.tsinghua.edu.cn/git/MNIST/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    ("https://mirrors.tuna.tsinghua.edu.cn/git/MNIST/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"),
    ("https://mirrors.tuna.tsinghua.edu.cn/git/MNIST/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
]

raw_dir = './data/MNIST/raw'
os.makedirs(raw_dir, exist_ok=True)

for url, filename in resources:
    gz_path = os.path.join(raw_dir, filename)
    raw_path = gz_path.replace('.gz', '')
    # 下载
    if not os.path.exists(gz_path) and not os.path.exists(raw_path):
        print(f"正在下载 {filename} ...")
        urllib.request.urlretrieve(url, gz_path)
    # 解压
    if os.path.exists(gz_path) and not os.path.exists(raw_path):
        print(f"正在解压 {filename} ...")
        with gzip.open(gz_path, 'rb') as f_in, open(raw_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"{filename} 解压完成！")
print("所有数据准备完毕！")

# ----------------------------------
# 1. 设置超参数和设备
# ----------------------------------

# Hyperparameters (超参数)
# 超参数是在开始学习过程之前设置值的参数，而不是通过训练得到的参数数据。
# 我们可以通过调整这些参数来影响模型的训练效果。
num_epochs = 10          # 训练的总轮数
batch_size = 64          # 每批次训练的样本数
learning_rate = 0.001    # 学习率，控制模型参数更新的步长

# Device Configuration (设备配置)
# 检查是否有可用的CUDA GPU，如果有，就使用GPU进行计算，否则使用CPU。
# 使用GPU可以大大加快训练速度。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"将使用 '{device}' 设备进行训练。")


# ----------------------------------
# 2. 准备 MNIST 数据集
# ----------------------------------

# 定义数据预处理的流程
# transforms.Compose(...) 会将一系列的转换操作串联起来。
transform = transforms.Compose([
    transforms.ToTensor(),  # 将 PIL Image 或 numpy.ndarray 转换为 FloatTensor，并将像素值从 [0, 255] 缩放到 [0.0, 1.0]
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化张量图像。参数分别是均值和标准差。
                                              # 这两个值是 MNIST 数据集的全局均值和标准差，使用它们可以获得更好的性能。
])

# 下载并加载训练数据集
# root='./data': 指定数据集下载后存放的目录。
# train=True: 表示加载的是训练集。
# download=True: 如果 './data' 目录下没有数据集，则自动下载。
# transform=transform: 应用我们上面定义好的预处理操作。
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transform,
                               download=True)

# 下载并加载测试数据集
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transform)

# 创建数据加载器 (Data Loader)
# DataLoader 是一个迭代器，可以方便地将数据集封装起来，实现批量（batch）读取数据。
# shuffle=True: 在每个 epoch 开始时，打乱训练集的数据顺序，这有助于提高模型的泛化能力。
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False) # 测试集不需要打乱顺序


# ----------------------------------
# 3. 构建 LeNet-5 模型
# ----------------------------------

# LeNet-5 经典结构:
# 输入 (1x28x28) ->
# C1: 卷积层 (6个5x5卷积核, padding=2) -> 输出 (6x28x28)
# S2: 平均池化层 (2x2) -> 输出 (6x14x14)
# C3: 卷积层 (16个5x5卷积核) -> 输出 (16x10x10)
# S4: 平均池化层 (2x2) -> 输出 (16x5x5)
# -> 展平 (Flatten) -> 16*5*5 = 400
# F5: 全连接层 (400 -> 120)
# F6: 全连接层 (120 -> 84)
# OUTPUT: 输出层 (84 -> 10)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 定义模型的卷积部分
        self.conv_part = nn.Sequential(
            # 第一个卷积层
            # 输入通道=1 (黑白图像), 输出通道=6, 卷积核大小=5x5, padding=2
            # 原始 LeNet 输入是 32x32, MNIST 是 28x28。加 padding=2 可以让 28x28 的输入经过 5x5 卷积核后尺寸不变 (28+2*2-5)/1 + 1 = 28
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), # 使用 ReLU 作为激活函数
            # 第一个池化层
            # 池化核大小=2x2, 步长=2
            nn.AvgPool2d(kernel_size=2, stride=2),

            # 第二个卷积层
            # 输入通道=6 (上一层的输出), 输出通道=16, 卷积核大小=5x5
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            # 第二个池化层
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # 定义模型的全连接部分
        self.fc_part = nn.Sequential(
            # 第一个全连接层
            # 16*5*5 是从卷积部分输出的特征图展平后的维度
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            # 第二个全连接层
            nn.Linear(120, 84),
            nn.ReLU(),
            # 输出层
            nn.Linear(84, 10) # 输出10个类别，分别对应数字0-9
        )

    # 定义前向传播的过程
    def forward(self, x):
        # 首先通过卷积部分
        x = self.conv_part(x)
        # 将多维的特征图展平为一维向量，以便输入到全连接层
        # x.size(0) 是 batch_size
        x = x.view(x.size(0), -1)
        # 然后通过全连接部分
        x = self.fc_part(x)
        return x

# 实例化模型，并将其移动到指定设备（GPU或CPU）
model = LeNet5().to(device)


# ----------------------------------
# 4. 定义损失函数和优化器
# ----------------------------------

# 损失函数 (Loss Function)
# nn.CrossEntropyLoss 用于多分类问题。它内部会自动对模型的输出执行 Softmax，然后计算交叉熵损失。
criterion = nn.CrossEntropyLoss()

# 优化器 (Optimizer)
# Adam 是一种常用的优化算法，它结合了 Momentum 和 RMSprop 的优点。
# model.parameters() 会告诉优化器需要更新模型中的哪些参数（权重和偏置）。
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ----------------------------------
# 5. 训练模型
# ----------------------------------

print("开始训练...")
total_step = len(train_loader)

# 外层循环是 epoch，表示完整地过一遍所有训练数据
for epoch in range(num_epochs):
    # 内层循环是 batch，每次处理一小批数据
    for i, (images, labels) in enumerate(train_loader):
        # 将图像和标签数据移动到指定的设备上
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播 (Forward pass)
        outputs = model(images) # 将图像输入模型，得到预测结果
        loss = criterion(outputs, labels) # 计算预测结果和真实标签之间的损失

        # 反向传播和优化 (Backward and optimize)
        optimizer.zero_grad() # 在计算梯度之前，将之前所有参数的梯度清零
        loss.backward()       # 计算当前损失对模型所有参数的梯度
        optimizer.step()      # 根据计算出的梯度，更新模型的参数

        # 每 100 个 batch 打印一次训练信息
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

print("训练完成！")


# ----------------------------------
# 6. 测试模型并计算准确率
# ----------------------------------

print("开始测试...")
# 将模型设置为评估模式 (evaluation mode)
# 这会关闭一些在训练时使用但在测试时不需要的功能，比如 Dropout
model.eval()

# 在测试阶段，我们不需要计算梯度，这样可以节省计算资源
with torch.no_grad():
    correct = 0
    total = 0
    # 遍历测试数据集
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 将测试图像输入模型
        outputs = model(images)
        
        # torch.max 会返回每一行（dim=1）的最大值和其索引。
        # 在这里，最大值就是模型的置信度，而索引就是预测的类别（数字0-9）。
        # 我们只关心预测的类别，所以用 _ 占位符忽略最大值。
        _, predicted = torch.max(outputs.data, 1)
        
        # 统计样本总数
        total += labels.size(0)
        # 统计预测正确的样本数
        correct += (predicted == labels).sum().item()

# 计算并打印最终的平均识别准确率
accuracy = 100 * correct / total
print(f'在 10000 张测试图像上的平均识别准确率: {accuracy:.2f} %')