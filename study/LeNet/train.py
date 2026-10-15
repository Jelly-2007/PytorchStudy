import torch
from torch import nn
from net import LeNet5
from torch.optim import lr_scheduler
from torchvision import datasets,transforms
import os

#所有路径都锚定到本脚本所在目录，与运行时的工作目录无关
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'save_model')

#数据转化为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

#加载训练数据集
train_dataset = datasets.MNIST(root=DATA_DIR,
                               train=True,
                               transform=data_transform,
                               download=True
                               )

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
#加载测试数据集
test_dataset = datasets.MNIST(root=DATA_DIR,
                               train=False,
                               transform=data_transform,
                               download=True
                               )

test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=16,shuffle=True)

#有显卡，可以转GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')
print(f'  torch 版本={torch.__version__}  编译时 CUDA={torch.version.cuda}')
if device.type == 'cuda':
    print(f'  显卡: {torch.cuda.get_device_name(0)}')
    print(f'  支持的架构: {torch.cuda.get_arch_list()}')
else:
    print('  ⚠ 未检测到可用 GPU，正在用 CPU 训练')

#调用net里面定义的模型,将模型数据转到GPU上
model = LeNet5().to(device)

#定义一个损失函数(交叉熵损失函数)
loss_fn = nn.CrossEntropyLoss()

#定义一个优化器
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)

#学习率每隔十轮，变为原来的0.1
lc_scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)

#定义训练函数
def train(dataloader,model,loss_fn,optimizer):
    loss,current,n=0.0, 0.0, 0
    for batch,(X,y) in enumerate(dataloader):
        #前向传播
        X,y = X.to(device),y.to(device)
        output = model(X)
        #计算损失
        cur_loss = loss_fn(output,y)
        _,pred = torch.max(output,axis=1)

        cur_acc = torch.sum(y == pred)/output.shape[0]

        #反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
    
    print("train_loss:" + str(loss/n))
    print("train_acc:" + str(current/n))

def val(dataloader, model, loss_fn):
    model.eval()
    loss,current,n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch,(X,y) in enumerate(dataloader):
            #前向传播
            X,y = X.to(device),y.to(device)
            output = model(X)
            #计算损失
            cur_loss = loss_fn(output,y)
            _,pred = torch.max(output,axis=1)
            cur_acc = torch.sum(y == pred)/output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print("val_loss:" + str(loss/n))
        print("val_acc:" + str(current/n))

        return current/n



#开始训练
epoch = 50

min_acc = 0

for t in range(epoch):
    print(f'epoch{t+1}\n---------------')
    train(train_dataloader,model,loss_fn,optimizer)
    a = val(test_dataloader,model,loss_fn)
    #每个epoch结束后更新学习率，使StepLR的衰减真正生效
    lc_scheduler.step()
    print(f'当前学习率: {optimizer.param_groups[0]["lr"]}')
    #保存最好的模型权重
    if a > min_acc:
        os.makedirs(MODEL_DIR, exist_ok=True)
        min_acc = a
        print('save best model')
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
print('Done!')

