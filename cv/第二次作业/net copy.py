import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import os

# 1. 检查PyTorch是否能找到CUDA
print(f"CUDA is available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 2. 查看可用的GPU数量
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # 3. 查看当前GPU的名称
    print(f"Current GPU name: {torch.cuda.get_device_name(0)}")

    # 4. 查看PyTorch编译时使用的CUDA版本
    print(f"PyTorch was built with CUDA version: {torch.version.cuda}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),#转为张量
    transforms.Normalize((0.1307,), (0.3081,))#归一化
])#预处理


#训练数据和测试数据
train_data = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)



class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(784,256)
        self.linear2 = torch.nn.Linear(256,64)
        self.linear3 = torch.nn.Linear(64,10)#十个手写数字的十个输出
    def forward(self,x):
        x = x.view(-1,784)#展平
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

model = Model().to(device)
criterion = torch.nn.CrossEntropyLoss()#交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)#SGD优化器

if os.path.exists('./model/model.pkl'):
    model.load_state_dict(torch.load('./model/model.pkl'))
    print('成功加载模型参数！')

def train(epoch):
    for index,data in enumerate(train_loader):
        input,target = data#获取数据
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()#梯度清零
        y_predict = model(input)
        loss = criterion(y_predict,target)
        loss.backward()#反向传播
        optimizer.step()#更新参数
        if index % 100 == 0:
            torch.save(model.state_dict(),'./model/model.pkl')#保存模型参数
            torch.save(optimizer.state_dict(),'./model/optimizer.pkl')#保存优化器参数
            print(loss.item())

def test():
    correct = 0 #正确的个数
    total = 0 #总的个数
    with torch.no_grad():#不计算梯度
        for data in test_loader:
            input,target = data
            input, target = input.to(device), target.to(device)
            output = model(input)
            pred = torch.argmax(output.data,dim=1)#获取最大值的索引
            total += target.size(0)
            correct += (pred==target).sum().item()
    print('测试集上的准确率为：%f %%' % (100 * correct / total))

if __name__ == '__main__':
    for i in range(10):
        print('第%d轮训练开始：' % (i+1))
        train(i)
        test()
