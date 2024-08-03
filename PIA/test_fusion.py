"""测试融合网络"""
import argparse
# -*- coding: utf-8 -*-


import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

# from data_loader.msrs_data import MSRS_data
# from models.cls_model import Illumination_classifier
# from models.common import gradient, clamp
# from models.fusion_model import PIAFusion
from drone_vehicle_dataset import DatasetDroneVehicle
from PIA.common import RGB2YCrCb, YCrCb2RGB, clamp, gradient
from PIA.model import PIAFusion
from PIA.cls_model import Illumination_classifier


def test():
    fusionmodel = PIAFusion()
    fusionmodel.load_state_dict(torch.load('./weight/fusion_model_epoch_29.pth'))
    # fusionmodel.load_state_dict(torch.load(r'D:\python_project\DroneDetect\PIA\weight\fusion_model_PIA_mini.pth'))
    fusionmodel.cuda()
    fusionmodel.eval()

    test_dataset = DatasetDroneVehicle()
    test_img = test_dataset.load_img(mode='val')
    batch_size = 4
    test_data = test_dataset.split_to_batch(test_img, batch_size=batch_size)
    print("the testing dataset is length:{}".format(len(test_data)))

    n_iter = len(test_data)

    # criteria_fusion = Fusionloss()
    cnt = 0
    with torch.no_grad():
        for it, image in tqdm(enumerate(test_data)):
            vis_image = image[:, :3].cuda()
            inf_image = image[:, [3]].cuda()
            vis_ycrcb_image = RGB2YCrCb(vis_image)
            vis_y_image = vis_ycrcb_image[:, [0]]
            cb = vis_ycrcb_image[:, [1]]
            cr = vis_ycrcb_image[:, [2]]

            fused_image = fusionmodel(vis_y_image, inf_image)
            # 强制约束范围在[0,1], 以免越界值使得最终生成的图像产生异常斑点
            fused_image = clamp(fused_image)

            for index in range(batch_size):
                # index = 2

                img = YCrCb2RGB(fused_image[index], cr[index], cb[index])
                img = transforms.ToPILImage()(img)
                file_name = test_dataset.val_img_files[cnt].split('\\')[-1]
                img.save(rf'D:\dataset\DroneVehicle\processed\val\fuse_img\{file_name}')
                cnt += 1
            # break

    # fusion_image = fusion_image[0].cpu().numpy() * 255
    # fusion_image = fusion_image.astype(np.uint8)

    # fusion_image = transforms.ToPILImage()(fusion_image)
    # print(fusion_image)
    # fusion_image = np.transpose(fusion_image, axes=(1, 2, 0))
    # img = PIL.Image.fromarray(fusion_image)
    # img.show()
    # fusion_image.show()


if __name__ == '__main__':
    test()
    exit()
    # parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    # parser.add_argument('--dataset_path', metavar='DIR', default='test_data/MSRS',
    #                     help='path to dataset (default: imagenet)')  # 测试数据存放位置
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
    #                     choices=['fusion_model'])
    # parser.add_argument('--save_path', default='results/fusion')  # 融合结果存放位置
    # parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
    #                     help='number of data loading workers (default: 4)')
    # parser.add_argument('--fusion_pretrained', default='pretrained/fusion_model_epoch_29.pth',
    #                     help='use cls pre-trained model')
    # parser.add_argument('--seed', default=0, type=int,
    #                     help='seed for initializing training. ')
    # parser.add_argument('--cuda', default=True, type=bool,
    #                     help='use GPU or not.')
    #
    # args = parser.parse_args()
    #
    # init_seeds(args.seed)
    #
    # test_dataset = MSRS_data(args.dataset_path)
    # test_loader = DataLoader(
    #     test_dataset, batch_size=1, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)
    #
    # if not os.path.exists(args.save_path):
    #     os.makedirs(args.save_path)
    #
    # # 如果是融合网络
    # if args.arch == 'fusion_model':
    #     model = PIAFusion()
    #     model = model.cuda()
    #     model.load_state_dict(torch.load(args.fusion_pretrained))
    #     model.eval()
    #     test_tqdm = tqdm(test_loader, total=len(test_loader))
    #     with torch.no_grad():
    #         for _, vis_y_image, cb, cr, inf_image, name in test_tqdm:
    #             vis_y_image = vis_y_image.cuda()
    #             cb = cb.cuda()
    #             cr = cr.cuda()
    #             inf_image = inf_image.cuda()
    #
    #             # 测试转为Ycbcr的数据再转换回来的输出效果，结果与原图一样，说明这两个函数是没有问题的。
    #             # t = YCbCr2RGB2(vis_y_image[0], cb[0], cr[0])
    #             # transforms.ToPILImage()(t).save(name[0])
    #
    #             fused_image = model(vis_y_image, inf_image)
    #             fused_image = clamp(fused_image)
    #
    #             rgb_fused_image = YCrCb2RGB(fused_image[0], cb[0], cr[0])
    #             rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
    #             rgb_fused_image.save(f'{args.save_path}/{name[0]}')
