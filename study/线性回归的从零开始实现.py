# ============================================================
# 线性回归的从零开始实现
# 目标:不使用 PyTorch 高层 API,手动实现线性回归的
#       数据生成 → 模型定义 → 损失函数 → 优化器 → 训练 全过程
# 线性模型:y = Xw + b,即用输入特征 X 的加权和(加偏置)预测 y
# ============================================================

import random       # 用于打乱数据(随机抽样)
import torch        # PyTorch 框架,提供张量(tensor)和自动求导

# ------------------------------------------------------------
# 1. 生成合成数据集
# ------------------------------------------------------------
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 的合成数据。

    参数:
        w             : 权重向量(真实值),形状 (特征数,)
        b             : 偏置(真实值),标量
        num_examples  : 样本数量
    返回:
        X             : 特征,形状 (num_examples, 特征数)
        y             : 标签,形状 (num_examples, 1)
    """
    # 生成服从标准正态分布 N(0,1) 的随机特征
    # 形状为 (num_examples, len(w)),即每行是一个样本、每列是一个特征
    X = torch.normal(0, 1, (num_examples, len(w)))

    # 矩阵乘法 X @ w,再广播加上偏置 b,得到"真实"的标签(不含噪声)
    y = torch.matmul(X, w) + b

    # 加上均值为 0、方差为 0.01 的高斯噪声,让数据更接近真实场景
    # y.shape 保证噪声张量与 y 形状一致
    y += torch.normal(0, 0.01, y.shape)

    # reshape((-1, 1)) 把 y 变成列向量 (num_examples, 1),方便后续算损失
    return X, y.reshape((-1, 1))


# 设定真实的模型参数(我们"假装"不知道,训练目标就是逼近它们)
true_w = torch.tensor([2.0, -3.0, 4.0])   # 真实权重,共 3 个特征
true_b = 4.2                               # 真实偏置

# 生成 1000 个样本
features, labels = synthetic_data(true_w, true_b, 1000)

# 打印第一个样本,直观看看生成的数据长什么样
print('features:', features[0], '\nlabel:', labels[0])

# ------------------------------------------------------------
# 2. 定义数据迭代器(小批量随机采样)
# ------------------------------------------------------------
def data_iter(batch_size, features, labels):
    """每次返回一个小批量 (batch) 的 (特征, 标签)。

    说明:训练时不是一次用全部数据,也不是一次只用一条,
    而是每次随机抽 batch_size 条,这样既稳定又高效。
    """
    num_examples = len(features)                     # 样本总数
    indices = list(range(num_examples))              # 生成索引列表 [0,1,2,...,n-1]

    random.shuffle(indices)                          # 随机打乱索引(每个 epoch 顺序都不同)
    for i in range(0, num_examples, batch_size):     # 按 batch_size 步长切片
        # 取当前批次的索引;min() 防止最后一批越界(可能不足 batch_size)
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)])
        # yield 让函数变成生成器,每调用一次吐出一批数据
        yield features[batch_indices], labels[batch_indices]


batch_size = 10   # 每个小批量 10 条样本

# 测试一下迭代器:打印第一批数据后 break,避免打印全部 100 批
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# ------------------------------------------------------------
# 3. 初始化模型参数
# ------------------------------------------------------------
# 权重 w:形状 (3, 1),从 N(0, 0.01) 随机初始化(接近 0 的小值)
# 偏置 b:初始化为 0
# requires_grad=True 表示:PyTorch 会跟踪对它们的运算,以便反向传播求梯度
w = torch.normal(0, 0.01, size=(3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# ------------------------------------------------------------
# 4. 定义模型
# ------------------------------------------------------------
def linreg(X, w, b):
    """线性回归模型:y_hat = X @ w + b"""
    return torch.matmul(X, w) + b

# ------------------------------------------------------------
# 5. 定义损失函数
# ------------------------------------------------------------
def squared_loss(y_hat, y):
    """平方损失(均方误差的一半):(y_hat - y)^2 / 2。

    说明:除以 2 是为了求导时系数抵消,梯度更简洁;
    这里没有取均值,均值会在 sgd 里通过 /batch_size 处理。
    """
    # y.reshape(y_hat.shape) 保证 y 的形状与预测值一致,才能相减
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# ------------------------------------------------------------
# 6. 定义优化器(小批量随机梯度下降 SGD)
# ------------------------------------------------------------
def sgd(params, lr, batch_size):
    """小批量随机梯度下降:沿梯度反方向更新参数。

    参数:
        params      : 需要更新的参数列表,如 [w, b]
        lr          : 学习率(步长)
        batch_size  : 批量大小
    """
    with torch.no_grad():          # 更新参数时不需要计算梯度,禁用梯度跟踪
        for param in params:
            # 梯度下降核心公式:param = param - lr * (梯度 / batch_size)
            # 除以 batch_size 是把整批的梯度和换算成平均梯度
            param -= lr * param.grad / batch_size
            # 清零梯度,否则下次 .backward() 会累加到旧梯度上
            param.grad.zero_()

# ------------------------------------------------------------
# 7. 训练
# ------------------------------------------------------------
lr = 0.03        # 学习率
num_epochs = 3   # 训练轮数(把全部数据过 3 遍)
net = linreg     # 给模型起个别名
loss = squared_loss   # 给损失函数起个别名

for epoch in range(num_epochs):                        # 外层循环:epoch(轮数)
    for X, y in data_iter(batch_size, features, labels):  # 内层循环:遍历小批量
        # 前向传播:用当前参数预测,再算损失(每个样本一个损失值)
        l = loss(net(X, w, b), y)
        # 反向传播:对 l.sum() 求梯度(必须求和成标量才能 backward)
        # 求得的梯度累加到 w.grad 和 b.grad 上
        l.sum().backward()
        # 用 SGD 更新一次参数
        sgd([w, b], lr, batch_size)

    # 每个 epoch 结束后,用整个训练集算一次平均损失,观察训练效果
    with torch.no_grad():          # 评估阶段不需要梯度,加快计算
        train_l = loss(net(features, w, b), labels)
        # float() 把 1x1 的张量转成 Python 浮点数,便于打印
        print(f"epoch {epoch + 1}, loss {float(train_l.mean())}")

# ------------------------------------------------------------
# 训练完成后,可以打印学到的参数与真实参数对比
# 理想情况下 w 应接近 [2.0, -3.0, 4.0],b 应接近 4.2
# ------------------------------------------------------------
print(f"\n真实参数: w = {true_w}, b = {true_b}")
print(f"学习到参数: w = {w.detach().flatten()}, b = {float(b.detach())}")
