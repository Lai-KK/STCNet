import torch
import SimpleITK as sitk
from datasets_swin import test_PRO_Loader_3D_position_16group
from SwinUNet_3D import TransUnet
import argparse
import yaml
import ml_collections
import time
import os
from thop import profile

def interpolate_3d(data, scale_factors):
    interpolated_data = zoom(data, scale_factors, order=3)
    return interpolated_data

def load_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    config = ml_collections.ConfigDict(config_dict)
    return config

def test_net(net, device, x_path, batch_size):
    # 加载训练集
    pro_dataset = test_PRO_Loader_3D_position_16group(x_path)
    print("数据个数：", len(pro_dataset))
    test_loader = torch.utils.data.DataLoader(dataset=pro_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 加载模型参数
    checkpoint_path=args.checkpoint+'/model'+str(args.model)+'.pth'
    checkpoint=torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    # 测试模式
    net.eval()
    # for x_data,y_data,name,encoder,slice in test_loader:
    for x_data,name,encoder,slice in test_loader:
        # 将数据拷贝到device中
        x_data = x_data.to(device=device, dtype=torch.float32)
        encoder = encoder.to(device=device, dtype=torch.float32)
        print(x_data.shape)
        print(encoder.shape)
        # 使用网络参数，输出预测结果
        with torch.no_grad():
            pred = net(x_data, encoder)

        # 保存pred
        img_3D = pred.cpu().detach().numpy()
        img_3D = img_3D * 200
        img_3D = sitk.GetImageFromArray(img_3D)
        name_str = name[0] if isinstance(name, tuple) else str(name)
        slice_num = slice.item() if torch.is_tensor(slice) else int(slice)
        # save_dir = os.path.join(args.output_path, f'test_{name_str}-{slice_num}.nii.gz')
        save_dir = os.path.join(args.output_path, f'test_{name_str}-0.nii.gz')
        sitk.WriteImage(img_3D, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('train parameters')
    parser.add_argument('--x_path', default=r'/Disk_16TB/heyutong/Swin/Sparse/x_data',
                        help='path to after_interplation data')
    #parser.add_argument('--y_path', default=r'/media/siat/Lenovo/Swin/swin_250/y_data',
                        #help='GT')
    parser.add_argument('--output_path', default=r'/Disk_16TB/heyutong/Swin/Sparse/results', help='path to output saves')
    parser.add_argument('--checkpoint', default=r'/home/heyutong/code/CIRS/checkpoint_TCIA', help='path to output saves')
    parser.add_argument('--model',  type=int,default=72, help='path to output saves')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--DataParallel', action='store_true', default=False)
    parser.add_argument('--device', type=int, default=2, help='device')

    args = parser.parse_args()

    # 选择设备，有cuda用cuda，没有就用cpu
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = os.environ['CUDA_VISIBLE_DEVICE'] = '1'
    device = torch.device('cuda', args.device)
    config = load_config('/home/heyutong/code/CIRS/config.yaml')

    net = TransUnet(config).to(device)

    # 开始前记录时间
    start_time = time.time()

    #test
    test_net(net, device, args.x_path,args.batch_size)

    # 结束后记录时间
    end_time = time.time()

    # 计算时间（单位：秒）
    training_time = (end_time - start_time)/3040
    print(f"耗时：{training_time:.2f} 秒")
