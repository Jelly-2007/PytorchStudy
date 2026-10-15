import torch
import os
from net import LeNet5
from torch.autograd import Variable
from torchvision import datasets,transforms
from torchvision.transforms import ToPILImage

#所有路径都锚定到本脚本所在目录，与运行时的工作目录无关
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#调用net里面定义的模型,将模型数据转到GPU上
model = LeNet5().to(device)

model.load_state_dict(torch.load("D:/Python项目/PytorchStudy/study/LeNet/save_model/best_model.pth"))

#获取结果
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",  
]

#把tensor转化为图片,方便可视化
show = ToPILImage()

#进入验证
for i in range(6,20):
    X, y = test_dataset[i][0],test_dataset[i][1]
    show(X).show()

    X = Variable(torch.unsqueeze(X,dim=0).float(),requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(X)

        predicted,actual = classes[torch.argmax(pred[0])],classes[y]

        print(f'predicted:"{predicted}",acutal:"{actual}"')
