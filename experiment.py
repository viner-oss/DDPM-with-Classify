import random
import numpy as np
import matplotlib.pyplot as plt
import torch

# 基本参数
T = 100
batch_size = 64
simulate_image = torch.randn([batch_size,3,32,32])
bat = random.randint(0, batch_size)

# time_step = torch.randint(0, T, (batch_size,)).long()
# label_step = torch.randint(0, 10, (batch_size, )).long()
#
# time_label_encoder = DDPM.Embedding_Util(10, T, batch_size)
# ddpm = DDPM.Diffusion(T)
#
# image, real_noise = ddpm.forward(simulate_image, time_step)
# image = time_label_encoder.time_embedding(image, time_step)
# image = time_label_encoder.label_embedding(image, label_step)
#
# model = DDPM.UNet()
# predict_noise, _ = model(image)

def Gaussian(x, mean, std):
    return (1/std*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/2*(std**2))



# # 从一个批次中随机选出一张图
# real_noise = real_noise.numpy()[bat,:,:,:]
# predict_noise = predict_noise.detach().numpy()[bat,:,:,:]

"""
显示预测噪声 与 真实噪声的概率密度曲线
训练阶段可以尝试使用
"""
def graph_show(real_np_noise, predict_np_noise):
    plt.figure(num=1, label='real || predict', figsize=(6,6))
    real_mean = np.mean(real_np_noise[:,:,:])
    predict_mean = np.mean(predict_np_noise[:,:,:])
    real_std = np.std(real_np_noise[:,:,:])
    predict_std = np.std(predict_np_noise[:,:,:])

    x_ch1 = np.linspace(-5, 5, 1000)
    y_real_ch1 = Gaussian(x_ch1, real_mean, real_std)
    y_predict_ch1 = Gaussian(x_ch1, predict_mean, predict_std)

    # 单独显示real
    plt.subplot(2,1,1)
    plt.title('real_noise')
    plt.plot(x_ch1, y_real_ch1)
    ax1 = plt.gca()
    ax1.spines['top'].set_color('none')
    ax1.spines['right'].set_color('none')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.spines['bottom'].set_position(('data', 0))
    ax1.spines['left'].set_position(('data', -5))
    plt.xticks(np.arange(-5, 5, 1))           # 修改x轴的精度
    plt.xlim(-5, 5)
    plt.yticks(np.arange(0, 5, 1))
    plt.ylim(0, 4)

    plt.scatter(real_mean, Gaussian(real_mean, mean=real_mean, std=real_std),
                s=50, c='red')
    plt.plot([real_mean, real_mean],
             [Gaussian(real_mean, mean=real_mean, std=real_std), 0],
             linestyle='--', color='k')

    plt.annotate(text='(%.4f,%.4f)' % (real_mean, Gaussian(real_mean, mean=real_mean, std=real_std)),
                 xy=(real_mean, Gaussian(real_mean, mean=real_mean, std=real_std)),
                 xytext=(+30, 0), xycoords='data', textcoords='offset points')

    # 单独显示predict
    plt.subplot(2,1,2)
    plt.title('predict_noise')
    plt.plot(x_ch1, y_predict_ch1)
    ax2 = plt.gca()
    ax2.spines['top'].set_color('none')
    ax2.spines['right'].set_color('none')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    ax2.spines['left'].set_position(('data', -5))
    ax2.spines['bottom'].set_position(('data', 0))
    plt.xticks(np.arange(-5, 5, 1))
    plt.xlim(-5 ,5)
    plt.yticks(np.arange(0, 5, 1))
    plt.ylim(0, 4)

    plt.scatter(predict_mean, Gaussian(predict_mean, mean=predict_mean, std=predict_std),
                s=50, c='red')
    plt.plot([predict_mean, predict_mean],
                [Gaussian(predict_mean, mean=predict_mean, std=predict_std), 0],
             linestyle='--', color='k')

    plt.annotate(text='(%.4f,%.4f)' %(predict_mean, Gaussian(predict_mean, mean=predict_mean, std=predict_std)),
                xy=(predict_mean, Gaussian(predict_mean, mean=predict_mean, std=predict_std)),
                 xytext=(+30,0), xycoords='data', textcoords='offset points')


    # 同时显示对比
    plt.figure(num=2, label='real && predict', figsize=(6,6))
    plt.subplot(1,1,1)
    plt.title('real && predict')
    plt.plot(x_ch1, y_real_ch1, label='real', color='blue')
    plt.plot(x_ch1, y_predict_ch1, label='predict', color='red')
    ax3 = plt.gca()
    ax3.spines['right'].set_color('none')
    ax3.spines['top'].set_color('none')
    ax3.xaxis.set_ticks_position('bottom')
    ax3.yaxis.set_ticks_position('left')
    ax3.spines['left'].set_position(('data', -5))
    ax3.spines['bottom'].set_position(('data', 0))
    plt.xticks(np.arange(-5, 5, 0.5))
    plt.xlim(-5, 5)
    plt.yticks(np.arange(0, 5, 1))
    plt.ylim(0, 4)

    plt.scatter(real_mean, Gaussian(real_mean, mean=real_mean, std=real_std),
                s=50, color='red')
    plt.plot([real_mean, real_mean],
             [Gaussian(real_mean, mean=real_mean, std=real_std), 0],
             linestyle='--', color='k', linewidth=1)
    plt.annotate(text='(%.4f,%.4f)' % (real_mean, Gaussian(real_mean, mean=real_mean, std=real_std)),
                 xy=(real_mean, Gaussian(real_mean, mean=real_mean, std=real_std)),
                 xytext=(-150, 0), xycoords='data', textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))

    plt.scatter(predict_mean, Gaussian(predict_mean, mean=predict_mean, std=predict_std),
                s=50, c='blue')
    plt.plot([predict_mean, predict_mean],
                [Gaussian(predict_mean, mean=predict_mean, std=predict_std), 0],
             linestyle='--', color='k', linewidth=1)

    plt.annotate(text='(%.4f,%.4f)' %(predict_mean, Gaussian(predict_mean, mean=predict_mean, std=predict_std)),
                xy=(predict_mean, Gaussian(predict_mean, mean=predict_mean, std=predict_std)),
                 xytext=(+60,0), xycoords='data', textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=.2'))
    plt.legend()

    plt.figure(num=3, label='image')
    plt.subplot(1,2,1)
    real_scale = (real_np_noise - np.min(real_np_noise)) / (np.max(real_np_noise) - np.min(real_np_noise)) * 255
    real_clip = np.clip(real_scale, 0, 255).astype(np.uint8)
    plt.imshow(np.swapaxes(np.swapaxes(real_clip[:,:,:], axis1=0, axis2=1), axis1=1, axis2=2), cmap='bone', interpolation='nearest', origin='upper')

    plt.subplot(1,2,2)
    predict_scale = (predict_np_noise - np.min(predict_np_noise)) / (np.max(predict_np_noise) - np.min(predict_np_noise)) * 255
    predict_clip = np.clip(predict_scale, 0, 255).astype(np.uint8)

    plt.imshow(np.swapaxes(np.swapaxes(predict_clip[:,:,:], axis1=0, axis2=1), axis1=1, axis2=2), cmap='bone', interpolation='nearest', origin='upper')

    plt.show()



"""
显示还原过程的图像历程
预测过程可以尝试使用
"""
def imshow_image(x, i):
    image_numpy = x[bat,:,:,:].numpy()
    image_scale = (image_numpy - np.min(image_numpy)) / (np.max(image_numpy) - np.min(image_numpy)) * 255
    image_clip = np.clip(image_scale, 0, 255).astype(np.uint8)

    plt.subplot(4,4,i)
    image = np.swapaxes(np.swapaxes(image_clip[:,:,:], axis1=0, axis2=1), axis1=1, axis2=2)
    plt.imshow(image, cmap='bone', interpolation='nearest', origin='upper')
    plt.imsave(f'D:\python\Segmentation\experiment\denoise_process{i}.png', arr=image)



if __name__ == "__main__":
    # graph_show(real_np_noise=real_noise, predict_np_noise=predict_noise)
    plt.figure(label='prediction')
    for i in range(5):
        imshow_image(simulate_image, i+1)

    plt.show()
