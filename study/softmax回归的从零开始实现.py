# ============================================================
# softmax 回归的从零开始实现(图像分类)
# 目标:不用 PyTorch 高层 API,手动实现 softmax 分类器,
#       对 Fashion-MNIST(10 类衣服图片)做多分类。
#
# 与线性回归的区别:线性回归输出 1 个实数(预测值);
# softmax 回归输出 10 个概率(每类一个),取概率最大的一类。
# ============================================================

import torch
from IPython import display
from d2l import torch as d2l

# ------------------------------------------------------------
# 1. 加载数据
# ------------------------------------------------------------
batch_size = 256
# d2l 库自带的加载函数:下载 Fashion-MNIST 并打包成 DataLoader
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 每张图是 28x28 的灰度图,拉平成一维向量 => 784 个输入特征
num_inputs = 784
# 10 个类别(衣服类型)
num_outputs = 10

# ------------------------------------------------------------
# 2. 初始化模型参数
# ------------------------------------------------------------
# W: (784, 10),每列对应一个类别的权重
# b: (10,),每个类别一个偏置
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# ------------------------------------------------------------
# 3. 定义 softmax 运算
# ------------------------------------------------------------
def softmax(X):
    """把一组实数转换成一组概率(和为 1,且每个都在 0~1 之间)。

    步骤:先 exp(把负数也变成正数) → 再除以每行的总和。
    """
    X_exp = torch.exp(X)                          # 对每个元素取指数
    partition = X_exp.sum(1, keepdim=True)        # 按行求和(keepdim 保持形状可广播)
    return X_exp / partition                       # 广播机制:每行除以该行的和

# ------------------------------------------------------------
# 4. 定义模型
# ------------------------------------------------------------
def net(X):
    """前向传播:拉平 → 线性变换 X@W+b → softmax 得到概率"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# ------------------------------------------------------------
# 5. 定义损失函数(交叉熵)
# ------------------------------------------------------------
def cross_entropy(y_hat, y):
    """交叉熵损失:-log(预测为真实类别的概率)。

    y_hat: 模型输出的概率,形状 (batch, 10)
    y     : 真实标签,形状 (batch,)
    """
    # 用高级索引取出每个样本在其真实类别上的预测概率
    return -torch.log(y_hat[range(len(y_hat)), y])

# ------------------------------------------------------------
# 6. 定义准确率等指标
# ------------------------------------------------------------
def accuracy(y_hat, y):
    """计算预测正确的样本数量"""
    # 如果 y_hat 是概率矩阵(二维),先取每行概率最大的类别
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    # 预测类别与真实类别逐元素比较,得到 0/1,再求和
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 一个简单的累加器,用于跨多个 batch 累加指标
class Accumulator:
    """在 n 个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    """计算模型在指定数据集上的准确率"""
    if isinstance(net, torch.nn.Module):
        net.eval()                       # 切换到评估模式(这里 net 不是 Module,会跳过)
    metric = Accumulator(2)              # 累加:正确数量、样本总数
    with torch.no_grad():                # 评估不需要梯度,省内存、更快
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]         # 正确数 / 总数 = 准确率

# ------------------------------------------------------------
# 7. 训练一个 epoch
# ------------------------------------------------------------
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练一个迭代周期,返回(平均训练损失, 训练准确率)"""
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)              # 累加:损失总和、正确数、样本总数

    # 这里 updater 是我们自己写的函数(不是 torch 优化器),走 else 分支
    if isinstance(updater, torch.optim.Optimizer):
        # 用 PyTorch 内置优化器的情况
        updater.zero_grad()
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
    else:
        # 用自定义优化器(我们自己写的 sgd)的情况
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            l.sum().backward()           # 反向传播求梯度
            updater(X.shape[0])          # 传入 batch_size 更新参数
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    return metric[0] / metric[2], metric[1] / metric[2]

# ------------------------------------------------------------
# 8. 绘图工具 Animator(画 loss/acc 随 epoch 变化的曲线)
# ------------------------------------------------------------
class Animator:
    """在训练过程中动态画图"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 用 lambda 捕获这些参数,便于 add 里重复调用
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """向图中添加一个数据点(x, y)并重绘"""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()                        # 清空旧图
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)          # 重画所有曲线
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

# ------------------------------------------------------------
# 9. 完整训练过程
# ------------------------------------------------------------
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型,num_epochs 轮"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    # 训练结束后打印最终结果
    print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}')

# ------------------------------------------------------------
# 10. 设置超参数并开始训练
# ------------------------------------------------------------
lr = 0.1                                  # 学习率

def updater(batch_size):
    """自定义更新函数:d2l.sgd 内部会做梯度下降并清零梯度"""
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 20
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

# ------------------------------------------------------------
# 11. 训练完成后,在测试集上可视化预测结果
# ------------------------------------------------------------
def predict_ch3(net, test_iter, n=6):
    """取测试集一批数据,展示图片 + 真实标签 + 预测标签"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)                       # 真实标签
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))   # 预测标签
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)

import matplotlib.pyplot as plt
plt.show()
