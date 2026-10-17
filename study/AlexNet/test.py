import torch
from net import AlexNet
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

#所有路径都锚定到本脚本所在目录，与运行时的工作目录无关
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'save_model')

ROOT_TRAIN = os.path.join(BASE_DIR, 'data', 'train')
ROOT_TEST = os.path.join(BASE_DIR, 'data', 'val')

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

train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST, transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'使用设备: {device}')

model = AlexNet().to(device)

#加载模型
model.load_state_dict(torch.load("D:/Python项目/PytorchStudy/study/AlexNet/save_model/best_model.pth"))

#获取预测结果
classes = [
    'cat',
    'dog'
]

#把tensor转化成照片格式
show = ToPILImage()

#进入到验证阶段
model.eval()
for i in range(400,450,2):
    X, y = val_dataset[i][0], val_dataset[i][1]
    show(X).show()
    X = Variable(torch.unsqueeze(X, dim=0).float(), requires_grad=True).to(device)
    with torch.no_grad():
        pred = model(X)
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'predicted:"{predicted},actual:"{actual}')