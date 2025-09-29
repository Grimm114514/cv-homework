import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
import os

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 输入通道1，输出通道6，卷积核5x5，padding=2
    nn.ReLU(),#激活函数
    nn.AvgPool2d(kernel_size=2, stride=2),  # 池化层，池化核2x2，步幅2
    nn.Conv2d(6, 16, kernel_size=5),  # 输入通道6，输出通道16，卷积核5x5
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # 池化层，池化核2x2，步幅2
    nn.Flatten(),  # 展平层，将多维输入展平成一维
    nn.Linear(16 * 5 * 5, 120),  # 全连接层，输入特征数16*5*5，输出特征数120
    nn.ReLU(),
    nn.Linear(120, 84),  # 全连接层，输入特征数
    nn.ReLU(),
    nn.Linear(84, 10)  # 全连接层，输入特征数
)

transform = transforms.Compose([
    transforms.ToTensor(),#转为张量
    transforms.Normalize((0.1307,), (0.3081,))#归一化
])#预处理

batch_size = 64
train_data = MNIST(root='./data', train=True, transform=transform, download=True)
train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = MNIST(root='./data', train=False, transform=transform, download=True)
test_iter = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()#交叉熵损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)#SGD优化器

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

if os.path.exists('./model2/model.pkl'):#加载模型
    net.load_state_dict(torch.load('./model2/model.pkl'))
    print('成功加载模型参数！')

def train(epoch):
    for index,data in enumerate(train_iter):
        input,target = data#获取数据
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()#梯度清零
        y_predict = net(input)
        loss = criterion(y_predict,target)
        loss.backward()#反向传播
        optimizer.step()#更新参数
        if index % 100 == 0:
            torch.save(net.state_dict(),'./model2/model.pkl')#保存模型参数
            torch.save(optimizer.state_dict(),'./model2/optimizer.pkl')#保存优化器参数
            print(loss.item())

def test():
    correct = 0 #正确的个数
    total = 0 #总的个数
    with torch.no_grad():#不计算梯度
        for data in test_iter:
            input,target = data
            input, target = input.to(device), target.to(device)
            output = net(input)
            pred = torch.argmax(output.data,dim=1)#获取最大值的索引
            total += target.size(0)
            correct += (pred==target).sum().item()
    print('测试集上的准确率为：%f %%' % (100 * correct / total))

if __name__ == '__main__':
    if not os.path.exists('./model2'):
        os.makedirs('./model2')
    for i in range(5):#训练5轮
        print('第%d轮训练开始：' % (i+1))
        train(i)
        test()

