import math
import torch
from torch import nn
from torch.nn import MultiheadAttention

import experiment

# 设定参数 数据参数
batch_size = 64
channel = 3
height = 32
width = 32
T = 100  # 总时间步数
beta_begin = 0.0001
beta_end = 0.02
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedding_Util:
    def __init__(self, num_classes:int, sum_timestep:int, dim:int, device):
        self.num_classes = num_classes
        self.sun_timestep = sum_timestep
        self.dim = dim
        self.device = device
        label_position = torch.arange(self.num_classes).unsqueeze(1).float()
        time_position = torch.arange(self.sun_timestep).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, self.dim, 2).float() * -(math.log(10000.0) / self.dim))

        label_embeddings = torch.zeros((self.num_classes, self.dim))
        label_embeddings[:, 0::2] = torch.sin(label_position * div)
        label_embeddings[:, 1::2] = torch.cos(label_position * div)
        self.label_embeddings = label_embeddings.to(device)

        time_embeddings = torch.zeros((self.sun_timestep, self.dim))
        time_embeddings[:, 0::2] = torch.sin(time_position * div)
        time_embeddings[:, 1::2] = torch.cos(time_position * div)
        self.time_embeddings = time_embeddings.to(device)

    def label_embedding(self, x, label):
        label_encoder = self.label_embeddings[label]
        in_channel = label_encoder.shape[1]
        out_channel = channel
        fc = nn.Linear(in_features=in_channel, out_features=out_channel).to(self.device)
        label_vector = fc(label_encoder)
        mapping_label_embedding = label_vector.unsqueeze(-1).unsqueeze(-1)
        mapping_label_embedding = mapping_label_embedding.expand(batch_size, -1, height, width)
        combine = torch.cat([x, mapping_label_embedding], dim=1)
        return combine

    def time_embedding(self, x, time_step):
        time_encoder = self.time_embeddings[time_step]
        in_channels = time_encoder.shape[1]
        out_channels = channel
        fc = nn.Linear(in_features=in_channels, out_features=out_channels).to(self.device)
        time_vector = fc(time_encoder)
        mapping_time_embedding = time_vector.unsqueeze(-1).unsqueeze(-1)
        mapping_time_embedding = mapping_time_embedding.expand(batch_size, -1, height, width)
        combine = torch.cat([x, mapping_time_embedding], dim=1)
        return combine

# 跨层连接的第一步
def skipped_connect(x, encode):
    return torch.cat([x,encode], dim=1)

# 生成二维图像的绝对位置编码  返回值是编码后的图像和新得到的通道数 维度必须是偶数 与最开始输入通道的关系是 2*dim + label_channels + time_channels + image_channels
class Position_Embedding(nn.Module):
    def __init__(self, dim:int, expansion_factor, device):
        super().__init__()
        self.expansion_factor = expansion_factor
        self.dim = dim
        self.hidden_dim = dim*2
        self.device = device
        self.div = torch.exp(torch.arange(0,dim,2).float() * -(math.log(10000.0)/dim)).unsqueeze(0).unsqueeze(0)

    def forward(self,X):
        batches, _, height, width = X.shape
        channels = channel
        x_embedding = torch.zeros([self.dim, height, width], dtype=torch.float).to(self.device)
        y_embedding = torch.zeros([self.dim, height, width], dtype=torch.float).to(self.device)

        height = torch.arange(0, height, dtype=torch.float)
        width = torch.arange(0, width, dtype=torch.float)
        x, y = torch.meshgrid(height, width)
        x = x.unsqueeze(2)
        y = y.unsqueeze(2)

        x_even = torch.sin(x * self.div).permute((2, 1, 0))
        x_odd = torch.cos(x * self.div).permute((2, 1, 0))
        y_even = torch.sin(y * self.div).permute((2, 1, 0))
        y_odd = torch.cos(y * self.div).permute((2, 1, 0))
        x_embedding[0::2, :, :] = x_even
        x_embedding[1::2, :, :] = x_odd
        y_embedding[0::2, :, :] = y_even
        y_embedding[1::2, :, :] = y_odd

        embedding = torch.cat([x_embedding, y_embedding], dim=0).unsqueeze(0).expand([batches, -1, -1, -1]).to(device=device)
        Y = torch.cat([embedding, X], dim=1)
        # mbconv = MBConv(Y.shape[1], channels, self.expansion_factor, false).to(device)
        # Y = mbconv(Y)

        return Y

"""
github_SA
"""
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

# 在跨层连接后 加入自注意力机制
"""
!!!分辨率过高时 记得切割图像 
"""
class SelfAttention(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, in_channels:int, device):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device

        self.W_K = nn.Linear(in_features=self.in_channels, out_features=embed_dim).to(device=device)
        self.W_Q = nn.Linear(in_features=self.in_channels, out_features=embed_dim).to(device=device)
        self.W_V = nn.Linear(in_features=self.in_channels, out_features=embed_dim).to(device=device)
        self.multihead = MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)

    def forward(self, x):
        batch, channels, height, width = x.shape
        image_vector = x.view([batch, channels, height*width]).permute([2,0,1])
        Q = self.W_Q(image_vector)
        K = self.W_K(image_vector)
        V = self.W_V(image_vector)
        attn_output, _ = self.multihead(Q,K,V)
        attn_output = (attn_output.view([height, width, batch, self.embed_dim])
                             .permute([2, 3, 0, 1]))
        return attn_output

# 加噪过程中的参数选择
class Diffusion:
    def __init__(self, num_time_steps:int, device, beta_src:float=0.0001, beta_cls:float=0.02):
        self.time_step = num_time_steps
        self.beta_src = beta_src
        self.beta_cls = beta_cls
        self.device = device

        # 对信号保留率α  噪声权值β  做提前处理  这里选择线性调度
        self.betas = torch.linspace(beta_src, beta_cls, steps=T, dtype=torch.float)      # β[]
        self.alphas = 1 - self.betas      # α[]
        self.cumpord_alphas = torch.cumprod(self.alphas,dim=0)        # 对特定时间步α的累乘
        self.sqrt_one_minus_cumpord_alphas = torch.sqrt(1-self.cumpord_alphas)
        self.sqrt_cumpord_alphas = torch.sqrt(self.alphas)
        self.time_label_encoder = Embedding_Util(10, T, batch_size, device)
        self.Position_encoder = Position_Embedding(dim=2, expansion_factor=3, device=device)

    def forward(self, x_0, time_step):
        noise = torch.randn_like(x_0)
        batch_sqrt_cumpord_alphas = self.sqrt_cumpord_alphas[time_step].view([x_0.shape[0],1,1,1]).to(self.device)
        batch_sqrt_one_minus_cumpord_alphas = self.sqrt_one_minus_cumpord_alphas[time_step].view([x_0.shape[0],1,1,1]).to(self.device)

        return (x_0*batch_sqrt_cumpord_alphas+noise*batch_sqrt_one_minus_cumpord_alphas,
                noise)

    # 返回值是生成的图像
    def prediction(self, model, time_step, epcho, test_step, x, appointed_label, flag:bool):
        with torch.no_grad():
            step = math.ceil(time_step / 16)
            sub = 1
            for i in reversed(range(time_step)):
                one_divide_sqrt_alpha = 1 / torch.sqrt(self.alphas[i])
                one_minus_alpha = 1 - self.alphas[i]
                sqrt_one_minus_cumpord_alpha = self.sqrt_one_minus_cumpord_alphas[i]
                sqrt_betas = torch.sqrt(self.betas[i])
                input_img = self.time_label_encoder.time_embedding(x, torch.full((batch_size,),i))
                # input_img = self.time_label_encoder.label_embedding(input_img, torch.full((batch_size,),appointed_label))   # 是否需要类别标签辅助生成
                input_img = self.Position_encoder(input_img).to(device)

                predict_noise = model(input_img)
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = 0

                x = one_divide_sqrt_alpha*(x - ((one_minus_alpha / sqrt_one_minus_cumpord_alpha) * predict_noise)) + sqrt_betas * noise

                if i % step == 0 and flag:
                    experiment.imshow_image(x, sub, epcho, test_step)
                    sub += 1

        return x

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
    def __init__(self, in_channels:int, out_channels:int, expansion_factor:int, num_heads:int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_factor = expansion_factor

        self.self_attention = SelfAttention(embed_dim=out_channels, num_heads=num_heads,
                                            in_channels=in_channels, device=device)
        self.SA = Self_Attn(in_dim=in_channels, activation='relu')

        self.mbconv = MBConv(in_channels=self.in_channels, out_channels=self.out_channels,
                             expansion_factor=self.expansion_factor, res=False)

    def forward(self, x, encode):
        y = skipped_connect(x, encode)

        y = self.self_attention(y)      # 多头注意力机制

        # y = self.SA(y)                # 单头注意力机制
        # y = self.mbconv(y)
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
        self.n_conv1 = Mobile_Conv(10,32)
        self.n_conv2 = Mobile_Conv(32, 64)

        # 下采样过程
        self.encoder_layer1 = Encode(64,128,1)
        self.encoder_layer2 = Encode(128, 256, 3)
        self.encoder_layer3 = Encode(256, 512, 3)
        self.encoder_layer4 = Encode(512, 1024, 6)

        #  上采样过程
        self.up_sample1 = Up_Sample(1024,512)
        self.decoder1 = Decoder(1024,512,6, 8)
        self.up_sample2 = Up_Sample(512,256)
        self.decoder2 = Decoder(512,256,3, 8)
        self.up_sample3 = Up_Sample(256, 128)
        self.decoder3 = Decoder( 256, 128, 3, 8)
        self.up_sample4 = Up_Sample(128, 64)
        self.decoder4 = Decoder( 128, 64, 1, 8)

        # 输出层
        self.output_conv1 = nn.Conv2d(in_channels=64, out_channels=32,
                                     kernel_size=3, stride=1, padding=1)
        self.output_BN_1 = nn.BatchNorm2d(num_features=32)
        self.output_relu = nn.ReLU6()
        self.output_conv2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1,
                                      stride=1, padding=0)
        self.output_BN_2 = nn.BatchNorm2d(num_features=3)

        self.classify = Classifier(3*32*32, 10)

        self.Adap_conv = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=1, stride=1,
                                   padding=0)

    def forward(self, x):
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

        return predict_noise


if __name__ == "__main__":
    input = torch.rand([batch_size,3,32,32]).to(device)        # 模拟图像数据
    time_step = torch.randint(0, T, (batch_size,)).long()      # 随机产生时间步
    label_step = torch.randint(0, 10, (batch_size, )).long()


    # 加噪过程 需要设置信号保有率α 噪声缩放因子β
    ddpm = Diffusion(T, device)        # 生成一个前向传播类 用于加噪和设置信号保有率参数
    input, real_noise = ddpm.forward(input, time_step)      # 对一整个批次同时加噪 并且时间步不同


    """
    时间编码与类编编码
    """
    Encoder = Embedding_Util(10, T, batch_size, device)
    Position_encoder = Position_Embedding(dim=2, expansion_factor=3, device=device)

    input = Encoder.time_embedding(input, time_step)
    # input = Encoder.label_embedding(input, label_step)    # 是否需要利用类别标签辅助生成
    input = Position_encoder(input)

    model = UNet().to(device)
    predict_noise = model(input)

