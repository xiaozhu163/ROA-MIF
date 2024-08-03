#!/usr/bin/python
# -*- encoding: utf-8 -*-
import datetime
import os
import time

import torch
from torch.autograd import Variable

from sea.loss import Fusionloss
from sea.model import FusionNet
from drone_vehicle_dataset import DatasetDroneVehicle
from sea.utils import RGB2YCrCb, YCrCb2RGB


def train_fusion(ui, model, dataset, optim, img_size, epochs, batch_size, learning_rate):
    if model == 'PSFusion':
        fusionmodel = FusionNet(output=1)
    else:
        fusionmodel = None

    fusionmodel.cuda()
    fusionmodel.train()

    batch_size = int(batch_size)
    epochs = int(epochs)
    img_size = int(img_size)

    lr_start = float(learning_rate)

    if optim == 'SGD':
        optimizer = torch.optim.SGD(fusionmodel.parameters(), lr=lr_start)
    elif optim == 'ADAM':
        optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    else:
        raise NotImplementedError

    train_dataset = DatasetDroneVehicle(ui=ui)
    train_img = train_dataset.load_img(mode='train', mini=True)
    train_data = train_dataset.split_to_batch(train_img, batch_size=8)
    ui.emit("the training dataset is length:{}\n".format(len(train_data)))
    # ui.repaint()

    n_iter = len(train_data)

    criteria_fusion = Fusionloss()

    st = glob_st = time.time()
    ui.emit('start training...' + '\n')
    # ui.repaint()
    for epo in range(0, epochs):
        # print('\n| epo #%s begin...' % epo)
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
        for it, image in enumerate(train_data):
            fusionmodel.train()
            image_vis = image[:, :3].cuda()
            image_ir = image[:, [3]].cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            logits = fusionmodel(image_vis_ycrcb, image_ir)

            optimizer.zero_grad()

            # fusion loss
            loss_fusion, loss_in, loss_grad = criteria_fusion(
                image_vis_ycrcb, image_ir, logits
            )

            # fusion_ycrcb = torch.cat(
            #     (logits, image_vis_ycrcb[:, 1:2, :, :],
            #      image_vis_ycrcb[:, 2:, :, :]),
            #     dim=1,
            # )
            # fusion_image = YCrCb2RGB(fusion_ycrcb)

            loss_total = loss_fusion
            loss_total.backward()

            optimizer.step()

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = n_iter * epo + it + 1
            eta = int((n_iter * epochs - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            # if now_it % 1 == 0:
            msg = ', '.join(
                [
                    'step: {it}/{max_it}',
                    'loss_total: {loss_total:.3f}',
                    'loss_in: {loss_in:.3f}',
                    'loss_grad: {loss_grad:.3f}',
                    '剩余时间: {eta}',
                ]
            ).format(
                it=now_it,
                max_it=n_iter * epochs,
                loss_total=loss_total.item(),
                loss_in=loss_in.item(),
                loss_grad=loss_grad.item(),
                eta=eta,
            )
            ui.emit(msg + '\n')
            # ui.repaint()
            print(msg)

            st = ed

    ui.emit('训练完成，保存模型中...\n')

    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")


    fusion_model_file = f'./weight/{dataset}/fusion_model_{now}.pth'
    torch.save(fusionmodel.state_dict(), fusion_model_file)
    # 记录模型的超参数

    file_name = './runs/' + now + ".txt"
    with open(file_name, 'w+') as w:
        s = '\n'.join(
            [
                '训练时间: {now}',
                '',
                '融合算法: {model}',
                '数据集: {dataset}',
                '优化器: {optim}',
                '初始学习率: {learning_rate}',
                'Epochs: {epochs}',
                'batch size: {batch_size}',
                '输入图像大小: {img_size}',

            ]
        ).format(
            now=now,
            model=model,
            dataset='可见光-红外' if dataset == 'IR' else '可见光-SAR',
            optim=optim,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size
        )
        w.write(s)
    ui.emit('保存完成')


if __name__ == "__main__":
    train_fusion()
