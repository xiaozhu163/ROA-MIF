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

# from data_loader.msrs_data import MSRS_data
# from models.cls_model import Illumination_classifier
# from models.common import gradient, clamp
# from models.fusion_model import PIAFusion
from drone_vehicle_dataset import DatasetDroneVehicle
from PIA.common import RGB2YCrCb, YCrCb2RGB, clamp, gradient
from PIA.model import PIAFusion
from PIA.cls_model import Illumination_classifier


def train():
    train_dataset = DatasetDroneVehicle()
    train_img = train_dataset.load_img(mode='train', mini=True)
    train_data = train_dataset.split_to_batch(train_img, batch_size=2)
    n_batch = len(train_data)

    lr = 0.001
    epochs = 10
    loss_weight = [3, 7, 50]

    model = PIAFusion()
    model = model.cuda()

    # 加载预训练的分类模型
    # one-hot标签[白天概率，夜晚概率]
    cls_model = Illumination_classifier(input_channels=3)
    cls_model.load_state_dict(torch.load(r'D:\python_project\DroneDetect\PIA\weight\best_cls.pth'))
    cls_model = cls_model.cuda()
    cls_model.eval()

    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        if epoch < epochs // 2:
            lr = lr
        else:
            lr = lr * (epochs - epoch) / (epochs - epochs // 2)

        # 修改学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        for i, image in enumerate(train_data):
            # vis_y_image = vis_y_image.cuda()
            # vis_image = vis_image.cuda()
            # inf_image = inf_image.cuda()

            vis_image = image[:, :3].cuda()
            inf_image = image[:, [3]].cuda()
            vis_ycrcb_image = RGB2YCrCb(vis_image)
            vis_y_image = vis_ycrcb_image[:, [0]]

            optimizer.zero_grad()
            fused_image = model(vis_y_image, inf_image)
            # 强制约束范围在[0,1], 以免越界值使得最终生成的图像产生异常斑点
            fused_image = clamp(fused_image)

            # 使用预训练的分类模型，得到可见光图片属于白天还是夜晚的概率
            pred = cls_model(vis_image)
            day_p = pred[:, 0]
            night_p = pred[:, 1]
            vis_weight = day_p / (day_p + night_p)
            inf_weight = 1 - vis_weight

            # pixel l1_loss
            loss_illum = F.l1_loss(inf_weight[:, None, None, None] * fused_image,
                                   inf_weight[:, None, None, None] * inf_image) + F.l1_loss(
                vis_weight[:, None, None, None] * fused_image,
                vis_weight[:, None, None, None] * vis_y_image)

            # auxiliary intensity loss
            loss_aux = F.l1_loss(fused_image, torch.max(vis_y_image, inf_image))

            # gradient loss
            gradinet_loss = F.l1_loss(gradient(fused_image), torch.max(gradient(inf_image), gradient(vis_y_image)))
            t1, t2, t3 = loss_weight
            loss = t1 * loss_illum + t2 * loss_aux + t3 * gradinet_loss

            print(f"[Epoch {epoch}/{epochs}] [{i}/{n_batch}] loss illum: {t1 * loss_illum.item():.3f} " +
                  f"loss aux: {t2 * loss_aux.item():.3f} gradinet_loss: {t3 * gradinet_loss.item():.3f} " +
                  f"Loss: {loss.item():.3f}")
            # print(epoch=epoch,
            #       loss_illum=t1 * loss_illum.item(),
            #       loss_aux=t2 * loss_aux.item(),
            #       gradinet_loss=t3 * gradinet_loss.item(),
            #       loss_total=loss.item())
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), f'./weight/fusion_model_PIA_mini.pth')


if __name__ == '__main__':
    train()
