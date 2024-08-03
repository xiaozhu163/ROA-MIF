import matplotlib.pyplot as plt
import numpy as np
import torch

# from build_label import build_label_LEVIR
# from dataset import process_LEVIR
from yolo.loss import Loss
from yolo.model import YOLOv5s
from PIL import Image, ImageDraw


class Visualize():
    def __init__(self):
        self.grid_pixel = (8, 16, 32)  # 每个grid的宽、高为几个像素
        self.number_anchor = 3
        self.nms_threshold = 0.3
        self.anchors = torch.Tensor(([(24, 34), (46, 21), (27, 57)],
                                     [(61, 28), (56, 40), (39, 63)],
                                     [(62, 58), (49, 131), (105, 77)]))
        self.class_dict = {0: "car", 1: "truck", 2: "bus", 3: 'van', 4: 'freight car'}

    def visual_label(self, data, label: list[torch.Tensor]):
        plt.figure()
        # plt.subplot(1, 2, 2)
        bbox_list = self.feature_map2bbox(label, is_pred=False)
        self.draw_img_and_bbox(data, bbox_list, is_pred=False)

    def visual_pred(self, data, pred_list: list[torch.Tensor]):
        plt.figure()
        # plt.subplot(1, 2, 1)
        bbox_list = self.feature_map2bbox(pred_list, is_pred=True)
        self.draw_img_and_bbox(data, bbox_list, is_pred=True)

    def draw_img_and_bbox(self, data, bbox: list[tuple], is_pred):
        if isinstance(data, torch.Tensor):
            # data = data * self.std + self.mean
            data = data.numpy()
            data = np.transpose(data, (1, 2, 0)) * 255
            data = data.astype(np.uint8)
            if data.shape[2] == 1:
                data = np.repeat(data, 3, axis=2)

        plt.imshow(data)

        for obj_cxywh in bbox:
            obj, c, x, y, w, h = obj_cxywh
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, color='red', fill=False))
            # if x > 480:
            #     x -= 55
            # if y < 20:
            #     y += 15
            if is_pred:  # 如果是预测值需要给出概率
                plt.text(x, y, self.class_dict[int(c)] + f" {obj:.2f}", size=10, color=(0, 1, 0))
            else:  # 是标签的话不要写出概率
                plt.text(x, y, self.class_dict[int(c)], size=10, color=(0, 1, 0))
        # 保存图片
        # plt.axis('off')
        # plt.xticks([])  # 去刻度
        # plt.savefig(f'../show/{is_pred}.jpg', bbox_inches='tight', pad_inches=0, dpi=300)

    def feature_map2bbox(self, feature_list, is_pred):
        if is_pred:
            feature_list = [i.permute((1, 2, 0)) for i in feature_list]

        bbox_list = []

        for feature_i in range(3):
            feature = feature_list[feature_i]

            for i in range(self.number_anchor):
                if is_pred:
                    positive_sample_index = feature[..., i * 10] > 0.
                else:
                    positive_sample_index = feature[..., i * 10] > 0.5

                # 正样本在feature_map中的第几行第几列
                grid_index = torch.nonzero(positive_sample_index)
                grid_index[:, [0, 1]] = grid_index[:, [1, 0]]  # 两列互换，因为第一列是横坐标，而grid_index本来的第一列是行数

                if not torch.any(positive_sample_index):
                    # 如果该anchor在所有的图片中都没有正样本，则跳过
                    continue

                # 选出正样本
                p = feature[positive_sample_index]
                p = p[:, i * 10:(i + 1) * 10]  # 第一个anchor则选第一个8列，第2个anchor选第二个8列......

                p_obj = p[:, [0]]
                p_cls = p[:, [1, 2, 3, 4, 5]]
                p_xy = p[:, [6, 7]]
                p_wh = p[:, [8, 9]]

                # obj需要经过sigmoid，作为概率
                p_obj = torch.sigmoid(p_obj)

                # 求出类别
                p_cls = torch.max(p_cls, dim=1)[1].view(-1, 1)

                if is_pred:
                    # 预测框回归
                    p_xy = torch.sigmoid(p_xy) * 2 - 0.5
                    p_wh = (torch.sigmoid(p_wh) * 2) ** 2 * self.anchors[feature_i, i]

                p_xy *= self.grid_pixel[feature_i]
                p_xy += grid_index * self.grid_pixel[feature_i]
                p_wh *= self.grid_pixel[feature_i]

                p_xy -= p_wh / 2  # 中心店xy坐标变为左上角的左边
                # p_xy[:, 1] -= 20  # 图片高600时上下各padding了20，这里减去

                p_obj_cxywh = torch.cat((p_obj, p_cls, p_xy, p_wh), dim=1)

                bbox_list.extend(p_obj_cxywh.numpy().tolist())

        # 变为tuple后去重
        bbox_list = [tuple(i) for i in bbox_list]
        bbox_list = list(set(bbox_list))

        if is_pred and len(bbox_list):
            # NMS处理。是预测值（非标签），且bbox_list不为空的时候才进行
            bbox_list = self.nms(bbox_list, threshold=self.nms_threshold)
        return bbox_list

    def nms(self, bbox_list, threshold):
        bbox_list = sorted(bbox_list, key=lambda x: x[0], reverse=True)

        bbox_nms = []
        while True:
            current = bbox_list[0]
            bbox_nms.append(current)

            bbox_list = [i for i in bbox_list if self.iou(current, i) < threshold]
            if not bbox_list:
                break

        return bbox_nms

    def iou(self, box1, box2):
        """
        参数都为obj,class,x,y,w,h的格式
        x,y为左上角的坐标
        """
        _, _, b1_x1, b1_y1, w1, h1 = box1
        _, _, b2_x1, b2_y1, w2, h2 = box2
        b1_x2, b1_y2 = b1_x1 + w1, b1_y1 + h1
        b2_x2, b2_y2 = b2_x1 + w2, b2_y1 + h2

        # Intersection area
        inter = max((min(b1_x2, b2_x2) - max(b1_x1, b2_x1)), 0) * \
                max((min(b1_y2, b2_y2) - max(b1_y1, b2_y1)), 0)

        # Union Area
        union = w1 * h1 + w2 * h2 - inter + 1e-7

        # IoU
        iou = inter / union

        return iou


if __name__ == '__main__':
    test_img = process_LEVIR(batch_size=128, train=False)
    test_label = build_label_LEVIR(batch_size=128, train=False)
    visual = Visualize()
    n_batch = len(test_img)

    model = YOLOv5s()
    model.load_state_dict(torch.load('./weight/checkpoint_350.pth'))
    model = model.to('cuda')

    criterion = Loss()

    model.eval()
    with torch.no_grad():
        for i in range(n_batch):
            data = test_img[i]
            label = test_label[i]

            data = data.to('cuda')
            label = [i.to('cuda') for i in label]

            outputs: torch.Tensor = model(data)
            break

            loss, obj_loss, class_loss, bbox_loss = criterion(outputs, label)
            print(f'[Batch {i + 1}/{n_batch}] Loss:{loss.item():.4f}'
                  f'|||obj:{obj_loss:.3f} cls:{class_loss:.3f} box:{bbox_loss:.3f}')
    # index=19，25，38
    for index in range(15, 25):
        pred = [i.cpu() for i in outputs]
        preds = [pred[0][index], pred[1][index], pred[2][index]]
        l = [i.cpu() for i in label]
        l = [l[0][index], l[1][index], l[2][index]]
        visual.visual_pred(data[index].cpu(), preds)
        visual.visual_label(data[index].cpu(), l)
    plt.show()
