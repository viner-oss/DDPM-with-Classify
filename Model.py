import torchvision
import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

import DDPM

"""
基本参数设置
"""
batch_size = 64
height = 32
width = 32
T = 100            # 总时间步数
beta_begin = 0.0001
beta_end = 0.02
epcho = 100        # 训练轮数

"""
时间步编码和位置编码
"""
add_noise = DDPM.Diffusion_Forward(num_time_steps=T)       # 添加前向传播类 用于设置α β 添加噪声
time_encoder = DDPM.Time_Embedding(T, batch_size)       # 时间步编码
label_encoder = DDPM.Label_Embedding(10, batch_size)

# 图像预处理
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
transform = torchvision.transforms.Compose([
        transforms.RandomHorizontalFlip(),       # 水平翻转（概率0.5）
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
])

# 准备训练集
train_set = torchvision.datasets.CIFAR10("../data",train=True,transform=transform,
                                         download=True)

# 准备测试集
test_set = torchvision.datasets.CIFAR10("../data",train=False,transform=transform,
                                         download=True)
# 加载数据集
train_dataloader = DataLoader(train_set,batch_size=batch_size)
test_dataloader = DataLoader(test_set,batch_size=batch_size)
# 获取数据集图片个数
train_set_size = len(train_set)
test_set_size = len(test_set)
print(f"训练数据集的长度:{train_set_size}")
print(f"测试数据集的长度:{test_set_size}")

# 学习率设置
learning_rate_min = 0.00002
learning_rate_max = 0.0001

# 模型 损失函数 优化器 余弦退火
model = DDPM.UNet()
noise_loss = MSELoss()
label_loss = CrossEntropyLoss()
optimizer = AdamW(model.parameters(),lr=learning_rate_max)
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=epcho,             # 训练轮数
    eta_min=learning_rate_min
)

# 迁移学习
pretrained_model = torchvision.models.mobilenet_v2(pretrained=True)


# 日志记录
writer = SummaryWriter("DDPM_SEG")

total_train_step = 0        # 每训练完一个patch 结果+1

for i in range(epcho):
    print(f"-----train {i+1} begin-----")
    for data in train_dataloader:
        img, label = data
        time_step = torch.randint(0, T, (batch_size,)).long()  # 随机产生时间步
        time_embedding = time_encoder(time_step)  # 从编码取出每个独立样本对应的时间步 用于嵌入样本

        label_embedding = label_encoder(label)

        real_noise = add_noise.noising_adding(img, time_step)
        input_img = DDPM.Time_Embedding_with_Pic_utils(real_noise, time_embedding)
        input_img = DDPM.Label_embedding_with_Pic_util(input_img, label_embedding)

        predict_noise, predict_label = model(input_img)

        loss_noise = noise_loss(predict_noise, input_img)   # 计算噪声损失值
        loss_classify = label_loss(predict_label, label)    # 计算类别损失值
        loss = 0.6*loss_noise + 0.4*loss_classify

        # 优化器优化模型
        optimizer.zero_grad()  # 优化清零
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 1 == 0:
            print(f"训练次数为{total_train_step}时,损失值是{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)



