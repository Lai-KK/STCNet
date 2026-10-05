from torch import optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from datasets_swin import PRO_Loader_3D_position_16group,test_PRO_Loader_3D_position_16group
from SwinUNet_3D import TransUnet
import SimpleITK as sitk
import argparse
import os
import pandas as pd
from datetime import datetime
import yaml
import ml_collections
from thop import profile
import math
import torch.distributed as dist

#加载config文件
def load_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    config = ml_collections.ConfigDict(config_dict)
    return config

#SSIM loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    # 获取批次大小和通道数
    batch_size, channel, _, _ = img1.size()
    
    # 创建窗口
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim_loss(pred, target):
    return 1 - ssim(pred, target)

# # 定义一个梯度计算函数，使用Sobel算子来计算图像的梯度
def sobel_gradient(image):
    # Sobel算子，计算x方向和y方向的梯度
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)
    
    return grad_x, grad_y

def gradient_consistency_loss(output_image):
    # 初始化损失
    loss = 0.0
    
    # 获取图像的形状
    num_slices, height, width = output_image.shape
    
    # 遍历每个切片（从第2个切片到倒数第2个切片）
    for i in range(1, num_slices - 1):
        # 当前切片及其相邻切片
        current_slice = output_image[i, :, :].unsqueeze(0).to(device)
        # print(current_slice.shape)
        prev_slice = output_image[i - 1, :, :].unsqueeze(0).to(device)
        next_slice = output_image[i + 1, :, :].unsqueeze(0).to(device)
        
        # 计算当前切片和相邻切片的梯度
        grad_current_x, grad_current_y = sobel_gradient(current_slice)
        grad_prev_x, grad_prev_y =sobel_gradient(prev_slice)
        grad_next_x, grad_next_y =sobel_gradient(next_slice)
        
        # 计算梯度差异
        grad_diff_prev = torch.abs(grad_current_x - grad_prev_x) + torch.abs(grad_current_y - grad_prev_y)
        grad_diff_next = torch.abs(grad_current_x - grad_next_x) + torch.abs(grad_current_y - grad_next_y)
        
        # 将这些差异累积到损失中
        # loss += torch.mean(grad_diff_prev) + torch.mean(grad_diff_next)
        loss += grad_diff_prev + grad_diff_next
    
    # 返回平均梯度一致性损失
    return loss / (num_slices - 2)

#TV损失
def tv_loss(image1, image2):
    """
    计算两个图像之间的TV损失
    :param image1: 形状为 (batch_size, channels, height, width) 的4D张量
    :param image2: 形状为 (batch_size, channels, height, width) 的4D张量
    :return: TV损失值
    """
    # 计算相邻切片之间的差异
    diff = image1 - image2
    
    # 计算横向和纵向的差异
    x_diff = diff[:, 1:, :] - diff[:, :-1, :]
    y_diff = diff[:, :, 1:] - diff[:, :, :-1]
    
    # 计算TV损失
    loss = torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))
    
    return loss

def total_variation_regularization_loss(output_image):
    """
    计算output_image的TV正则化损失
    :param output_image: 形状为 (batch_size, slices, height, width) 的4D张量
    :return: 总TV正则化损失值
    """
    # 初始化总损失
    total_loss = 0.0
    
    # 获取output_image的切片数
    num_slices, height, width = output_image.shape
    
    # 遍历每个切片（从第2个切片到倒数第2个切片）
    for i in range(1, num_slices - 1):
        # 当前切片及其相邻切片
        current_slice = output_image[i, :, :].unsqueeze(1)
        prev_slice = output_image[i - 1, :, :].unsqueeze(1)
        next_slice = output_image[i + 1, :, :].unsqueeze(1)
        
        # 计算当前切片与相邻切片之间的TV损失
        tv_loss_prev = tv_loss(current_slice, prev_slice)
        tv_loss_next = tv_loss(current_slice, next_slice)
        
        # 将这些差异累积到总损失中
        total_loss += tv_loss_prev + tv_loss_next
    
    # 返回平均TV正则化损失
    return total_loss / (num_slices - 2)


def train_net(net, device, x_path,y_path,val_path, epochs, batch_size):
    # 加载训练集
    pro_dataset = PRO_Loader_3D_position_16group(x_path,y_path)
    val_dataset = test_PRO_Loader_3D_position_16group(val_path,y_path)
    train_loader = torch.utils.data.DataLoader(dataset=pro_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    # 定义初始学习率
    lr_0 = 0.0001
    total_epochs = args.epochs  # 总训练 epoch 数
    decay_start = 25  # 何时开始衰减

    # 定义RMSprop算法
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # optimizer = optim.Adam(net.parameters(), lr=0.00001)
    optimizer = optim.Adam(net.parameters(),lr=lr_0,weight_decay=0.0001)
    # optimizer = optim.Adam(net.parameters(),lr=0.00001,weight_decay=0.0001)
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义学习率调度策略
    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0  # 前 50 轮保持不变
        else:
            return max(0.0, 1.0 - (epoch - decay_start) / (total_epochs - decay_start))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 定义Loss算法
    criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    # best_loss = float('inf')
    '''resume'''
    if args.resume:
        # FM = '/home/siat/hyt/SwinUnet/500_FM/checkpoint_FM'
        checkpoint_file = os.path.join(args.checkpoint, f'model{args.epoch}.pth')
        # checkpoint_file = os.path.join(FM, f'model{args.epoch}.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        # print(checkpoint)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        net.load_state_dict(checkpoint['net'])
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        epoch, init_epoch = 0, 0

    # 训练epochs次
    for epoch in range(init_epoch, args.epochs + 1):
        time='%s'%datetime.now() #获取当前时间
        print('epoch：' + str(epoch))
        '''训练模式'''
        net.train()
        total_train_loss = 0.0
        train_batches = 0
        for x_data,y_data,encoder in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            x_data = x_data.to(device=device, dtype=torch.float32)
            y_data = y_data.to(device=device, dtype=torch.float32)
            encoder = encoder.to(device=device, dtype=torch.float32)
            pred = net(x_data,encoder)

            # 计算loss
            pred = pred.reshape(pred.shape[2], pred.shape[3], pred.shape[4])
            y_data = y_data.reshape(y_data.shape[1], y_data.shape[2], y_data.shape[3])
            loss_MSE = criterion(pred, y_data)
            loss_SSIM = ssim_loss(pred.unsqueeze(0), y_data.unsqueeze(0))
            #梯度一致性损失
            loss_grad = gradient_consistency_loss(pred)
            #TV loss
            loss_TV = total_variation_regularization_loss(pred)
            # total loss
            a,b,c=args.a,args.b,args.c
            # train_loss = '%f' % loss + a*loss_grad + b* loss_TV
            total_loss = loss_MSE + a*loss_grad+ b* loss_TV + c * loss_SSIM
            total_loss = total_loss.mean()
            # train_loss = '%f' % torch.mean(total_loss).item()
            train_loss = '%f' % total_loss

            total_train_loss += total_loss.item()
            train_batches += 1

            #存储
            # list=[epoch,time,train_loss]
            # 更新参数
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()  # 更新学习率

        # 计算平均训练损失
        avg_train_loss = total_train_loss / train_batches
        print(f"Epoch {epoch + 1}: Learning Rate = {scheduler.get_last_lr()[0]:.6f}")
        print(f'Epoch [{epoch}/{epochs}], Average Training Loss: {avg_train_loss:.4f}')

        # 保存网络参数
        if (epoch) % 1 == 0:
            # print(f'Epoch [{epoch}/{epochs}],Loss: {total_loss.item():.4f}')
            content = {'epoch': epoch + 1,
                       'net': net.state_dict(),
                       'loss':total_loss}
            checkpoint_path = args.checkpoint + '/model' + str(epoch) + '.pth'
            torch.save(content, checkpoint_path)

            checkpoint = torch.load(checkpoint_path, map_location=device)
            net.load_state_dict(checkpoint['net'])
            '''验证模式'''
            net.eval()
            total_val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for x_data,y_data,name,encoder,slice in val_loader:
                    x_data = x_data.to(device=device, dtype=torch.float32)
                    y_data = y_data.to(device=device, dtype=torch.float32)
                    encoder = encoder.to(device=device, dtype=torch.float32)

                    pred = net(x_data, encoder)
                    
                    # 计算验证损失
                    pred = pred.reshape(pred.shape[2], pred.shape[3], pred.shape[4])
                    y_data = y_data.reshape(y_data.shape[1], y_data.shape[2], y_data.shape[3])
                    val_loss_MSE = criterion(pred, y_data)
                    val_loss_SSIM = ssim_loss(pred.unsqueeze(0), y_data.unsqueeze(0))
                    val_loss_grad = gradient_consistency_loss(pred)
                    # val_loss_TV = total_variation_regularization_loss(pred)

                    val_total_loss = val_loss_MSE + a*val_loss_grad  + c*val_loss_SSIM
                    total_val_loss += val_total_loss.mean().item()
                    val_batches += 1

                    # 保存预测结果
                    img_3D = pred.cpu().detach().numpy()
                    img_3D = img_3D*200
                    img_3D = sitk.GetImageFromArray(img_3D)
                    name_str = name[0] if isinstance(name, tuple) else str(name)
                    slice_num = slice.item() if torch.is_tensor(slice) else int(slice)
                    save_dir = os.path.join(args.output_path, f'epoch{epoch}_{name_str}-{slice_num}.nii.gz')
                    sitk.WriteImage(img_3D, save_dir)

            # 计算平均验证损失
            avg_val_loss = total_val_loss / val_batches
            print(f'Epoch [{epoch}/{epochs}], Average Validation Loss: {avg_val_loss:.4f}')

        # 保存训练和验证损失
        list = [epoch, time, avg_train_loss, avg_val_loss]
        data = pd.DataFrame([list])
        data.to_csv(os.path.join(args.csv_save, 'train_CIRS.csv'), mode='a', header=False, index=False)

        print(f'Epoch [{epoch}/{epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('train parameters')
    parser.add_argument('--x_path', default=r'/home/siat/hyt/SwinUnet/CIRS/16_group',
                        help='path to train data')
    parser.add_argument('--y_path', default=r'/home/siat/hyt/SwinUnet/CIRS/y_data',
                    help='path to GT data')
    parser.add_argument('--val_path', default=r'/home/siat/hyt/SwinUnet/CIRS/test',
                        help='path to before_interplation saves')
    parser.add_argument('--output_path', default=r'/home/siat/hyt/SwinUnet/CIRS/result', help='path to output saves')
    parser.add_argument('--checkpoint', default=r'/home/siat/hyt/SwinUnet/CIRS/checkpoint', help='path to output saves')
    parser.add_argument('--csv_save', default=r'',help='path to csv save')
    parser.add_argument('--config_path', default=r'', help='path to config')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--device', type=int, default=2, help='device')
    parser.add_argument('--epochs', type=int, default=100, help='number of epoch')
    parser.add_argument('--DataParallel', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--epoch', type=int, default=54, help='choose an epoch to retrain')
    parser.add_argument('--a', type=int, default=0.5, help='weight of loss_grad')
    parser.add_argument('--b', type=int, default=0, help='weight of loss_TV')
    parser.add_argument('--c', type=int, default=0.5, help='weight of loss_SSIM')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')


    args = parser.parse_args()

    # 初始化分布式训练环境
    dist.init_process_group(backend='nccl')

    # 选择设备，有cuda用cuda，没有就用cpu
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = os.environ['CUDA_VISIBLE_DEVICE'] = '0,1,2,3'
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    # device = torch.device('cuda', args.device)

    # gradient_model = Gradient_Net().to(device)
    
    config = load_config(os.path.join(args.config_path,'config.yaml'))
    net = TransUnet(config).to(device)

    # # 使用DataParallel封装模型
    # if args.DataParallel:
    #     if torch.cuda.device_count() > 1:
    #         print(f'Using {torch.cuda.device_count()} GPUs')
    #         net = nn.DataParallel(net)

    # 使用 DistributedDataParallel 进行多 GPU 并行计算
    if args.DataParallel and torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank)

    

    # '''保存loss'''
    # df = pd.DataFrame(columns=['time', 'train Loss'])
    # df.to_csv('./train_loss.csv', index=False)

    # 将网络拷贝到deivce中
    # net.to(device=device)
    # print(net.device)
    # print(next(net.parameters()).is_cuda)
    train_net(net, device, args.x_path,args.y_path,args.val_path, args.epochs, args.batch_size)