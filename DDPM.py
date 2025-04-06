import math
import torch
from torch import nn

# 设定参数 数据参数
batch_size = 64
height = 32
width = 32
T = 100  # 总时间步数
beta_begin = 0.0001
beta_end = 0.02

# 类别编码嵌入
def Label_embedding_with_Pic_util(x, label_embedding):
    in_channels = label_embedding.shape[1]
    out_channels = x.shape[1] // 2
    fc = nn.Linear(in_features=in_channels, out_features=out_channels)
    label_vector = fc(label_embedding)
    mapping_label_embedding = label_vector.unsqueeze(-1).unsqueeze(-1)
    mapping_label_embedding = mapping_label_embedding.expand(batch_size, -1, height, width)
    combine = torch.cat([x, mapping_label_embedding], dim=1)
    return combine

# 类别编码
class Label_Embedding(nn.Module):
    def __init__(self, num_classes:int, embedding_dim:int):
        super().__init__()
        self.dim = embedding_dim
        self.num_classes = num_classes
        positions = torch.arange(self.num_classes).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, self.dim, 2).float() * -(math.log(10000.0) / self.dim))
        embeddings = torch.zeros((self.num_classes, self.dim))
        embeddings[:, 0::2] = torch.sin(positions * div)
        embeddings[:, 1::2] = torch.cos(positions * div)
        self.embeddings = embeddings

    def forward(self, labels):
        embeds = self.embeddings[labels]
        return embeds


# 跨层连接的第一步
def skipped_connect(x, encode):
    return torch.cat([x,encode], dim=1)



# 时间步编码嵌入
def Time_Embedding_with_Pic_utils(x, time_embedding):
    in_channels = time_embedding.shape[1]
    out_channels = x.shape[1]
    fc = nn.Linear(in_features=in_channels,out_features=out_channels)
    time_vector = fc(time_embedding)
    mapping_time_embedding = time_vector.unsqueeze(-1).unsqueeze(-1)
    mapping_time_embedding = mapping_time_embedding.expand(batch_size,-1,height,width)
    combine = torch.cat([mapping_time_embedding,x],dim=1)
    return combine



# 生成时间步编码
class Time_Embedding(nn.Module):
    def __init__(self, Time_steps:int, dim:int):
        super(Time_Embedding, self).__init__()
        self.dim = dim
        self.Time_step = Time_steps
        positions = torch.arange(self.Time_step).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,self.dim,2).float() * -(math.log(10000.0)/self.dim))
        embeddings = torch.zeros((self.Time_step,self.dim))
        embeddings[:,0::2] = torch.sin(positions * div)
        embeddings[:,1::2] = torch.cos(positions * div)
        self.embeddings = embeddings

    def forward(self,t):
        embeds = self.embeddings[t]
        return embeds



# 生成绝对位置编码
class Position_Embedding(nn.Module):
    def __init__(self, patch_nums:int, embedding_dim:int):
        super().__init__()
        self.patch_nums = patch_nums
        self.embedding_dim = embedding_dim

        position = torch.arange(self.patch_nums).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,self.embedding_dim,2).float() * -(math.log(10000.0)/self.embedding_dim))
        self.embeddings = torch.zeros([self.patch_nums, self.embedding_dim])
        self.embeddings[:,0::2] = torch.sin(position * div)
        self.embeddings[:,1::2] = torch.cos(position * div)

    def forward(self,patch_index):
        return self.embeddings[patch_index]



# 在跨层连接后 加入自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, patch_size:int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.patches_size = patch_size

        self.W_q = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.W_k = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.W_v = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batches, channels, height, width = x.shape
        height_num = height // self.patches_size
        width_num = width // self.patches_size

        embedding = Position_Embedding(channels*height_num*width_num, self.patches_size**2)
        patches = []
        index = 0

        for bat in range(batches):
            for channel in range(channels):
                for i in range(height_num):
                    for j in range(width_num):
                        patch = x[bat, channel, i*self.patches_size:(i+1)*self.patches_size,
                                j*self.patches_size:(j+1)*self.patches_size]
                        patch = torch.flatten(patch)
                        patch += embedding(index)
                        patches.append(patch.unsqueeze(1).float())
                        index += 1
            index = 0

        new_tensor = torch.stack(patches)
        new_tensor = new_tensor.view([batches, channels, height_num * width_num,
                                      self.patches_size * self.patches_size]).permute([0, 1, 3, 2])

        Q = self.W_q(new_tensor)
        K = self.W_k(new_tensor)
        V = self.W_v(new_tensor)

        Attention_score = torch.matmul(K.transpose(-2, -1), Q) / self.hidden_dim ** 0.5
        Attention_pro = self.sigmoid(Attention_score)

        return torch.matmul(V, Attention_pro).view([batches, channels, height, width])



# 加噪过程中的参数选择
class Diffusion_Forward:
    def __init__(self, num_time_steps:int, beta_src:float=0.0001, beta_cls:float=0.02):
        self.time_step = num_time_steps
        self.beta_src = beta_src
        self.beta_cls = beta_cls

        # 对信号保留率α  噪声权值β  做提前处理  这里选择线性调度
        self.betas = torch.linspace(beta_src, beta_cls, steps=T, dtype=torch.float)      # β[]
        self.alphas = 1 - self.betas      # α[]
        self.cumpord_alphas = torch.cumprod(self.alphas,dim=0)        # 对特定时间步α的累乘
        self.sqrt_one_minus_cumpord_alphas = torch.sqrt(1-self.cumpord_alphas)
        self.sqrt_cumpord_alphas = torch.sqrt(self.alphas)

    def noising_adding(self, x_0, time_step):
        noise = torch.randn_like(x_0)
        batch_sqrt_cumpord_alphas = self.sqrt_cumpord_alphas[time_step].view([x_0.shape[0],1,1,1])
        batch_sqrt_one_minus_cumpord_alphas = self.sqrt_one_minus_cumpord_alphas[time_step].view([x_0.shape[0],1,1,1])

        return (x_0*batch_sqrt_cumpord_alphas+
                noise*batch_sqrt_one_minus_cumpord_alphas)



# U_net编码器部分的MBConv模块
class MBConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, expansion_factor:int, res:bool):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_channels = in_channels * expansion_factor
        self.res = res

        # 扩展层
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.expansion_channels,
                                     kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.expansion_channels),
            nn.ReLU6()
        )

        # 深度可分离卷积
        self.conv_dw = nn.Sequential(
            nn.Conv2d(in_channels=self.expansion_channels, out_channels=self.expansion_channels,
                                    kernel_size=3, stride=1, padding=1,
                                    groups=self.expansion_channels),
            nn.BatchNorm2d(self.expansion_channels),
            nn.ReLU6()
        )

        # 额外引入SE模块 作为通道注意力机制
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.SE = nn.Sequential(
            nn.Linear(in_features=self.expansion_channels,
                      out_features=self.expansion_channels//16),
            nn.ReLU6(),
            nn.Linear(in_features=self.expansion_channels//16,
                      out_features=self.expansion_channels),
            nn.Sigmoid()
        )

        # 投影层
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=self.expansion_channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU6()
        )

    def forward(self, x):
        y = self.expansion(x)
        y = self.conv_dw(y)
        batches, channels, _, _ = y.shape
        channel_attention = self.avgpool(y).view([batches, channels])
        channel_attention = self.SE(channel_attention).view([batches, channels, 1, 1])
        y = channel_attention.expand_as(y) * y
        y = self.projection(y)
        if self.res:
            y += x
        return y



# mobile_netV1的普通卷积 以适应初始操作
class Mobile_Conv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 通道数不增加的深度可分离卷积
        self.conv_keep = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                      kernel_size=3, stride=1, padding=1, groups=self.in_channels),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(),
            nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6()
        )

        # 通道数增加的深度可分离卷积
        self.conv_increase = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                      kernel_size=3, stride=1, padding=1, groups=self.in_channels),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU6()
        )

    def forward(self, x):
        y = self.conv_keep(x)
        y = self.conv_increase(y)
        return y



# 编码器部分
class Encode(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, expansion_factor:int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 通道数两次卷积一次增加
        self.mbconv_block1 = MBConv(in_channels=self.in_channels, out_channels=self.out_channels,
                                    expansion_factor=expansion_factor, res=False)
        self.mbconv_block2 = MBConv(in_channels=self.out_channels, out_channels=self.out_channels,
                                    expansion_factor=expansion_factor, res=True)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        y = self.maxpool(x)
        y = self.mbconv_block1(y)
        y = self.mbconv_block2(y)

        return y



# 上采样块
class Up_Sample(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up_sample = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                            kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        y = self.up_sample(x)
        return y


# 解码器
class Decoder(nn.Module):
    """
    input_dim是指每张特征图中patch的个数   hidden_dim通常情况下与input_dim数量相同
    因为在堆叠后通道数会加倍 所以还需要一次卷积来将通道数融合减半
    """
    def __init__(self, input_dim:int, hidden_dim:int, patch_size:int, in_channels:int, out_channels,
                 expansion_factor:int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_factor = expansion_factor

        self.self_attention = SelfAttention(input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                                            patch_size=self.patch_size)
        self.mbconv = MBConv(in_channels=self.in_channels, out_channels=self.out_channels,
                             expansion_factor=self.expansion_factor, res=False)

    def forward(self, x, encode):
        y = skipped_connect(x, encode)
        y = self.self_attention(y)
        y = self.mbconv(y)
        y += x
        return y

# 分类器
class Classifier(nn.Module):
    def __init__(self, in_dim:int, out_dim:int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.classify = nn.Sequential(
            nn.ReLU6(),
            nn.Linear(in_features=self.in_dim, out_features=self.out_dim),
            nn.Softmax(0)
        )

    def forward(self, x):
        y = x.view([x.shape[0], -1])
        y = self.classify(y)

        return y


# 主模型
"""
通道数和参数的修改在这里进行
"""
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_conv1 = Mobile_Conv(9,32)
        self.n_conv2 = Mobile_Conv(32, 64)

        # 下采样过程
        self.encoder_layer1 = Encode(64,128,1)
        self.encoder_layer2 = Encode(128, 256, 3)
        self.encoder_layer3 = Encode(256, 512, 3)
        self.encoder_layer4 = Encode(512, 1024, 6)

        #  上采样过程
        self.up_sample1 = Up_Sample(1024,512)
        self.decoder1 = Decoder(2*2,2*2,2,1024,512,6)
        self.up_sample2 = Up_Sample(512,256)
        self.decoder2 = Decoder(2*2,2*2,4,512,256,3)
        self.up_sample3 = Up_Sample(256, 128)
        self.decoder3 = Decoder(4*4, 4*4, 4, 256, 128, 3)
        self.up_sample4 = Up_Sample(128, 64)
        self.decoder4 = Decoder(4*4, 4*4, 8, 128, 64, 1)

        # 输出层
        self.output_conv1 = nn.Conv2d(in_channels=64, out_channels=32,
                                     kernel_size=3, stride=1, padding=1)
        self.output_BN_1 = nn.BatchNorm2d(num_features=32)
        self.output_relu = nn.ReLU6()
        self.output_conv2 = nn.Conv2d(in_channels=32, out_channels=9, kernel_size=1,
                                      stride=1, padding=0)
        self.output_BN_2 = nn.BatchNorm2d(num_features=9)

        self.classify = Classifier(9*32*32, 10)


    def forward(self, x):
        # 通道数扩展和匹配阶段
        y = self.n_conv1(x)

        # 编码器部分
        y = self.n_conv2(y)
        encode1 = y
        y = self.encoder_layer1(y)
        encode2 = y
        y = self.encoder_layer2(y)
        encode3 = y
        y = self.encoder_layer3(y)
        encode4 = y
        y = self.encoder_layer4(y)

        # 解码器部分
        y = self.up_sample1(y)
        y = self.decoder1(y, encode4)
        y = self.up_sample2(y)
        y = self.decoder2(y, encode3)
        y = self.up_sample3(y)
        y = self.decoder3(y, encode2)
        y = self.up_sample4(y)
        y = self.decoder4(y, encode1)

        # 通道数压缩和匹配阶段
        y = self.output_conv1(y)
        y = self.output_BN_1(y)
        y = self.output_relu(y)
        y = self.output_conv2(y)
        y = self.output_BN_2(y)

        predict_noise = y
        predict_label = self.classify(y)

        return predict_noise, predict_label



if __name__ == "__main__":
    input = torch.rand([batch_size,3,32,32])        # 模拟图像数据
    time_step = torch.randint(0, T, (batch_size,)).long()      # 随机产生时间步
    label_step = torch.randint(0, 10, (batch_size, )).long()

    # 加噪过程 需要设置信号保有率α 噪声缩放因子β
    add_noisy = Diffusion_Forward(T)        # 生成一个前向传播类 用于加噪和设置信号保有率参数
    input = add_noisy.noising_adding(input, time_step)      # 对一整个批次同时加噪 并且时间步不同

    """
        时间步编码助手 可以让所有时间步都有自己对应的编码
        可以通过时间步随机采样建立与时间编码的索引关系
    """
    Label_encoder = Label_Embedding(10, embedding_dim=batch_size)
    label_embedding = Label_encoder(label_step)

    Time_Embedding_util = Time_Embedding(Time_steps=T, dim=batch_size)
    time_embedding = Time_Embedding_util(time_step)          # 根据当前批次做时间步选择
    input = Time_Embedding_with_Pic_utils(input,time_embedding)     # 时间步编码与图像数据融合
    input = Label_embedding_with_Pic_util(input, label_embedding)

    model = UNet()
    predict_noise, predict_label = model(input)

    print(predict_label)
    print(predict_noise)
    # print(predict_noise.shape)
    # print(model)
    # print(model.state_dict())