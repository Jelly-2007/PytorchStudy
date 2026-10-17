import torch
from torch import nn
from net import AlexNet
import numpy as np
from torch.optim import lr_scheduler
import os

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#所有路径都锚定到本脚本所在目录，与运行时的工作目录无关
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'save_model')

ROOT_TRAIN = os.path.join(BASE_DIR, 'data', 'train')
ROOT_TEST = os.path.join(BASE_DIR, 'data', 'val')

NUM_WORKERS = 4

#将图像的像素值归一化到[-1,1]之间
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

#只创建 device 对象，不初始化 CUDA；放在模块级，train()/val() 可以直接使用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    model.train()#切换到训练模式，否则val()里的eval()会残留，导致dropout失效
    loss, current, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(dataloader):
        image, y = X.to(device), y.to(device)
        output = model(image)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred) / output.shape[0]

        #反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    train_loss = loss / n
    train_acc = current / n
    print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")
    return train_loss, train_acc

#定义验证函数
def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            image, y = X.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

    val_loss = loss / n
    val_acc = current / n
    print(f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
    return val_loss, val_acc

#定义画图函数
def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('训练集和验证集的loss值对比图')
    plt.show()


def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('训练集和验证集的acc值对比图')
    plt.show()


def main():
    #数据集和 DataLoader 必须在 main() 里创建，不能放在模块级
    train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
    val_dataset = ImageFolder(ROOT_TEST, transform=test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True,
                                  persistent_workers=NUM_WORKERS > 0)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True,
                                persistent_workers=NUM_WORKERS > 0)

    print(f'使用设备: {device}')
    print(f'  torch 版本={torch.__version__}  编译时 CUDA={torch.version.cuda}')
    if device.type == 'cuda':
        print(f'  显卡: {torch.cuda.get_device_name(0)}')
        print(f'  支持的架构: {torch.cuda.get_arch_list()}')
    else:
        print('  ⚠ 未检测到可用 GPU，正在用 CPU 训练')

    model = AlexNet().to(device)

    #定义一个损失函数
    loss_fn = nn.CrossEntropyLoss()

    #定义一个优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    #学习率每隔十轮变为原来的0.5
    #变量名改成 scheduler，避免在函数里和 import 进来的 lr_scheduler 模块重名
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    #开始训练
    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []

    #训练轮数
    epoch = 80
    min_acc = 0

    for t in range(epoch):
        print(f'epoch {t+1}\n---------------')
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
        val_loss, val_acc = val(val_dataloader, model, loss_fn)
        scheduler.step()
        loss_train.append(train_loss)
        acc_train.append(train_acc)
        loss_val.append(val_loss)
        acc_val.append(val_acc)

        #保存最好的模型权重
        if val_acc > min_acc:
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
            min_acc = val_acc
            print(f"save best model,第{t+1}轮")
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
        #保存最后一轮的权重文件
        if t == epoch - 1:
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'last_model.pth'))

    print("训练结束！")
    matplot_loss(loss_train, loss_val)
    matplot_acc(acc_train, acc_val)


if __name__ == '__main__':
    main()
