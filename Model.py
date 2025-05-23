import random
import torchvision
import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import DDPM
from DDPM import Embedding_Util
import experiment

"""
基本参数设置
"""
batch_size = 64
height = 32
width = 32
T = 100            # 总时间步数
beta_begin = 0.00002
beta_end = 0.0001
epcho = 100        # 训练轮数
num_classes = 10 # 类别个数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
时间步编码和位置编码
"""
ddpm = DDPM.Diffusion(num_time_steps=T, device=device)       # 添加前向传播类 用于设置α β 添加噪声
time_label_encoder = Embedding_Util(10, T, batch_size, device)
Position_encoder = DDPM.Position_Embedding(dim=2, expansion_factor=3, device=device)

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
                                         download=False)

# 准备测试集
test_set = torchvision.datasets.CIFAR10("../data",train=False,transform=transform,
                                         download=False)
# 加载数据集
train_dataloader = DataLoader(train_set,batch_size=batch_size, drop_last=True)
test_dataloader = DataLoader(test_set,batch_size=batch_size, drop_last=True)
# 获取数据集图片个数
train_set_size = len(train_set)
test_set_size = len(test_set)
print(f"训练数据集的长度:{train_set_size}")
print(f"测试数据集的长度:{test_set_size}")

# 学习率设置
learning_rate_min = 0.00002
learning_rate_max = 0.0001

# 模型 损失函数 优化器 余弦退火
model = DDPM.UNet().to(device)
noise_loss = MSELoss().to(device)
optimizer = AdamW(model.parameters(),lr=learning_rate_max)
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=epcho,             # 训练轮数
    eta_min=learning_rate_min
)

"""
need or no need pretrained_weight
"""
pretrained_dict = torch.load(r'C:\Users\lenovo\Desktop\DDPM\Parameters\DPM_parameters_120.pt')
model.load_state_dict(pretrained_dict)

# 日志记录
writer = SummaryWriter("DDPM_Multi_head_LOG")

total_train_step = 0        # 每训练完一个patch 结果+1DDPM_parameters.pt
total_test_step = 1         # 每预测完一个patch 结果+1

# train_stage 单独训练生成模型
for i in range(epcho):
     print(f"-----train {i+1} begin-----")
     for data in train_dataloader:
         img, label = data
         img = img.to(device)
         label = label.to(device)
         time_step = torch.randint(0, T, (batch_size,)).long()  # 随机产生时间步

         input_img, real_noise = ddpm.forward(img, time_step)
         input_img = time_label_encoder.time_embedding(input_img, time_step)
         # input_img = time_label_encoder.label_embedding(input_img, label)   # 是否需要加入类别标签辅助生成
         input_img = Position_encoder(input_img)

         predict_noise = model(input_img)

         loss_noise = noise_loss(predict_noise, real_noise)   # 计算噪声损失值
         loss = loss_noise

         # 优化器优化模型
         optimizer.zero_grad()  # 优化清零
         loss.backward()
         optimizer.step()

         total_train_step += 1
         if total_train_step % 100 == 0:
             print(f"训练次数为{total_train_step}时,损失值是{loss.item()}")
             writer.add_scalar("train_loss", loss.item(), total_train_step)
         torch.cuda.empty_cache()
         if total_train_step % 700 == 0:
             experiment.graph_show(real_noise, predict_noise, total_train_step)

     torch.save(model.state_dict(), fr'C:\Users\lenovo\Desktop\DDPM\Parameters/DPM_parameters_{i}.pt')

     # model.eval()
     # print(f"-----test {i+1} begin-----")
     # with torch.no_grad():
     #     for CNT in range(1,51):
     #         origin_noise = torch.randn([batch_size, 3, 32, 32]).to(device)
     #         t = T                    # 随机产生时间步
     #         if CNT % 5 == 0:
     #             predict_img = ddpm.prediction(model, t, CNT,
     #                        test_step=total_test_step, x=origin_noise, appointed_label=0, flag=True)
     #         else:
     #             predict_img = ddpm.prediction(model, t, CNT,
     #                        test_step=total_test_step, x=origin_noise, appointed_label=0, flag=False)
     #
     #         total_test_step += 1
     #
     #     print("image save!")

writer.close()


