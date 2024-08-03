#!/usr/bin/python
# -*- encoding: utf-8 -*-
import datetime
import os
import time
import numpy as np

import PIL.Image
import cv2
import torch
from torch.autograd import Variable

from loss import Fusionloss
from sea.model import FusionNet
from drone_vehicle_dataset import DatasetDroneVehicle
from sea.utils import RGB2YCrCb, YCrCb2RGB


def test():
    # num: control the segmodel
    fusionmodel = FusionNet(output=1)
    fusionmodel.load_state_dict(torch.load('./weight/fusion_model_mini.pth'))
    fusionmodel.cuda()
    fusionmodel.eval()

    test_dataset = DatasetDroneVehicle()
    test_img = test_dataset.load_img(mode='val')
    test_data = test_dataset.split_to_batch(test_img, batch_size=17)
    print("the testing dataset is length:{}".format(len(test_data)))

    n_iter = len(test_data)

    criteria_fusion = Fusionloss()

    with torch.no_grad():
        for it, image in enumerate(test_data):
            fusionmodel.eval()
            image_vis = image[:, :3].cuda()
            image_ir = image[:, [3]].cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            logits = fusionmodel(image_vis_ycrcb, image_ir)

            # fusion loss
            loss_fusion, loss_in, loss_grad = criteria_fusion(
                image_vis_ycrcb, image_ir, logits
            )

            fusion_ycrcb = torch.cat(
                (logits, image_vis_ycrcb[:, 1:2, :, :],
                 image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            break

            loss_total = loss_fusion

            msg = ', '.join(
                [
                    'loss_total: {loss_total:.4f}',
                    'loss_in: {loss_in:.4f}',
                    'loss_grad: {loss_grad:.4f}',
                ]
            ).format(
                max_it=n_iter,
                loss_total=loss_total.item(),
                loss_in=loss_in.item(),
                loss_grad=loss_grad.item(),
            )
            print(msg)

    print(fusion_image.shape)
    print(fusion_image[0])
    fusion_image = fusion_image[11].cpu().numpy() * 255
    fusion_image = fusion_image.astype(np.uint8)
    print(fusion_image)
    fusion_image = np.transpose(fusion_image, axes=(1, 2, 0))
    img = PIL.Image.fromarray(fusion_image)
    img.show()


if __name__ == "__main__":
    test()
