import torch
from torch import nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.grid_width = float(800 // 25)
        self.grid_height = float(600 // 25)
        self.lambda_class = 1
        self.lambda_bbox = 1
        self.number_anchor = 3  # 每个grid中anchor的数量
        self.eps = 1e-7

        self.anchors = torch.Tensor(([(24, 34), (46, 21), (27, 57)],
                                     [(61, 28), (56, 40), (39, 63)],
                                     [(62, 58), (49, 131), (105, 77)]))
        self.anchors = self.anchors.to('cuda')
        self.is_ciou = True
        self.focal_loss = FocalLoss(gamma=5.0, alpha=0.25)

    def forward(self, pred_list: tuple[torch.Tensor], label_list: list[torch.Tensor]):
        # todo anchor没有变为grid的倍数
        pred_list = [i.permute((0, 2, 3, 1)) for i in pred_list]

        obj_loss = torch.zeros(1, device='cuda')
        class_loss = torch.zeros(1, device='cuda')
        bbox_loss = torch.zeros(1, device='cuda')

        # 置信度损失
        # 按理说置信度损失要在下方的for循环内计算，但是为了方便直接在这计算了
        # for i in range(3):
        #     obj_loss += nn.BCEWithLogitsLoss()(pred_list[i][:, :, :, [0, 8, 16]],
        #                                        label_list[i][:, :, :, [0, 8, 16]])

        for feature_i in range(3):
            label = label_list[feature_i]
            pred = pred_list[feature_i]

            for i in range(self.number_anchor):
                # 根据置信度找到正样本，置信度所在的位置是[0,8,16,24...]等被8整除的位置，i*8表示这些位置
                positive_sample_index = label[..., i * 10] > 0.5
                positive_sample_index = positive_sample_index & (label[..., (i * 10) + 9] > 0.)
                positive_sample_index = positive_sample_index & (label[..., (i * 10) + 8] > 0.)
                negetive_sample_index = ~positive_sample_index

                if not torch.any(positive_sample_index):
                    # 如果该anchor在所有的图片中都没有正样本，则跳过
                    continue

                # 选出正样本
                p = pred[positive_sample_index]
                l = label[positive_sample_index]
                p = p[:, i * 10:(i + 1) * 10]  # 第一个anchor则选第一个8列，第2个anchor选第二个8列......
                l = l[:, i * 10:(i + 1) * 10]

                # 选出负样本
                p_neg = pred[negetive_sample_index]
                l_neg = label[negetive_sample_index]
                p_neg = p_neg[:, i * 10:(i + 1) * 10]
                l_neg = l_neg[:, i * 10:(i + 1) * 10]
                # 负样本只用来计算obj损失，所以只需要第0列
                p_neg_obj = p_neg[:, [0]]
                l_neg_obj = l_neg[:, [0]]

                # p_表示预测值，l_表示真实值
                p_obj = p[:, [0]]
                l_obj = l[:, [0]]
                p_cls = p[:, [1, 2, 3, 4, 5]]
                l_cls = l[:, [1, 2, 3, 4, 5]]
                p_xy = p[:, [6, 7]]
                p_wh = p[:, [8, 9]]
                l_xywh = l[:, [6, 7, 8, 9]]

                # 置信度损失
                obj_loss += self.focal_loss(p_obj, l_obj)  # 正样本置信度损失
                obj_loss += self.focal_loss(p_neg_obj, l_neg_obj)  # 负样本置信度损失
                # obj_loss += self.focal_loss(pred[:,[0]],label[:,[0]])

                # nn.CrossEntropyLoss()自带softmax步骤，不需要自己进行softmax操作
                # 类别损失
                class_loss += nn.CrossEntropyLoss()(p_cls, l_cls)

                # 预测框回归
                p_xy = torch.sigmoid(p_xy) * 2 - 0.5
                p_wh = (torch.sigmoid(p_wh) * 2) ** 2 * self.anchors[feature_i, i]
                p_xywh = torch.cat((p_xy, p_wh), dim=1)
                if self.is_ciou:
                    iou = self.ciou(p_xywh, l_xywh)
                else:
                    iou = self.iou(p_xywh, l_xywh)

                bbox_loss += (1.0 - iou).mean()
                if torch.isnan(bbox_loss):
                    assert 0, "出现了nan"

        total_loss = obj_loss + class_loss * self.lambda_class + bbox_loss * self.lambda_bbox
        return total_loss, obj_loss.item(), class_loss.item(), bbox_loss.item()

    def iou(self, pred: torch.Tensor, label: torch.Tensor):
        """
        参数都为x,y,w,h的格式
        :param pred:
        :param label:
        :return:
        """
        (x1, y1, w1, h1), (x2, y2, w2, h2) = pred.chunk(4, dim=1), label.chunk(4, dim=1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - inter + self.eps

        # IoU
        iou = inter / union

        return iou

    def ciou(self, pred: torch.Tensor, label: torch.Tensor, eps=1e-7):
        """
        参数都为x,y,w,h的格式
        :param pred:
        :param label:
        :return:
        """
        (x1, y1, w1, h1), (x2, y2, w2, h2) = pred.chunk(4, dim=1), label.chunk(4, dim=1)
        h1 = h1 + eps  # 防止在计算arctan(w1/h1)时，h1做分母报错
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - inter + self.eps

        # IoU
        iou = inter / union
        # CIoU
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (
                b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
        v = (4 / torch.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)  # CIoU


class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=0.25):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss()  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        return loss.mean()


if __name__ == '__main__':
    l = Loss()
