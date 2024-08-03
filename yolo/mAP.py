import glob
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class mAP_Calculator():
    def __init__(self, dataset='LEVIR'):
        self.dataset = dataset

    def label2csv(self, path=None):
        """
        将path文件夹下的所有txt文件合为一个csv文件，作为计算mAP的标注
        :param path: 文件夹路径
        :return:
        """
        if self.dataset == 'LEVIR':
            path = r'D:\dataset\DroneVehicle\processed/val/labelr'
            self.cls_name = {0: "car", 1: "truck", 2: "bus", 3: 'van', 4: 'freight car'}

        test_label_paths = glob.glob(path + '\\*.txt')

        data_list = []

        for label_file in test_label_paths:
            img_id = label_file.split('\\')[-1].replace('.txt', '.jpg')
            with open(label_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n').rstrip(' ')
                line = line.split(' ')
                line = [int(i) for i in line]
                cls, x1, y1, x2, y2 = line
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(799, x2)
                y2 = min(599, y2)
                cls = self.cls_name[cls]
                data_list.append((img_id, cls, x1, x2, y1, y2))

        data = pd.DataFrame(data_list, columns=['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax'])
        return data

    def cal_mAP(self, preds: pd.DataFrame, valid=None, show_PR_curve=False):
        if valid is None:
            valid = self.label2csv()
        ann_unique = valid['ImageID'].unique()
        unique_classes = valid['LabelName'].unique().astype(str)

        all_detections = self.get_detections(preds)
        all_annotations = self.get_real_annotations(valid)
        average_precisions = {}
        iou_threshold = 0.5

        for zz, label in enumerate(unique_classes):
            # Negative class
            if str(label) == 'nan':
                continue

            false_positives = []  # 每个值对应一个检测框，若第i个检测框为TP，则false_positives[1]为1否则0
            true_positives = []  # 每个值对应一个检测框，若第i个检测框为TP，则false_positives[1]为0否则1
            scores = []  # 每个检测框的置信度
            num_annotations = 0  # 每张图片总共的标注框(ground truth)个数
            for i in range(len(ann_unique)):  # 逐图片处理
                detections = []
                annotations = []
                id = ann_unique[i]
                if id in all_detections:
                    if label in all_detections[id]:
                        detections = all_detections[id][label]
                if id in all_annotations:
                    if label in all_annotations[id]:
                        annotations = all_annotations[id][label]
                num_annotations += len(annotations)
                annotations = np.array(annotations, dtype=np.float64)
                detected_annotations = []
                for d in detections:  # 获取label类，第i张图片的检测框d
                    scores.append(d[4])  # 获取检测框d的置信度

                    if len(annotations) == 0:  # 如果该图片没有标注信息，则该检测框肯定是FP
                        false_positives.append(1)
                        true_positives.append(0)
                        continue
                    # 计算检测框d与所有标注框的交并比，将结果存入overlaps中
                    overlaps = self.iou(np.expand_dims(np.array(d, dtype=np.float64), axis=0), annotations)
                    # 获取与检测框d交并比最大的那个标注框的索引
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    # 获取最大的交并比值
                    max_overlap = overlaps[0, assigned_annotation]
                    # class_iou.append(max_overlap)
                    # 如果交并比大于阈值，且与检测框d匹配的最佳标注框（assigned_annotation ） 还未被分配，
                    # 则检测框d为TP否则为FP
                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives.append(0)
                        true_positives.append(1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives.append(1)
                        true_positives.append(0)

            scores = np.array(scores)
            false_positives = np.array(false_positives)
            true_positives = np.array(true_positives)

            indices = np.argsort(-scores)  # -score而不是score是为了计算降序的顺序
            false_positives = false_positives[indices]  # 按照scores顺序对fals_positives排序
            true_positives = true_positives[indices]  # 按照scores顺序对true_positives排序
            false_positives_num = np.cumsum(false_positives)  # 计算置信度小于阈值的FP个数
            true_positives_num = np.cumsum(true_positives)  # 计算置信度小于阈值的TP个数
            recall = true_positives_num / num_annotations
            precision = true_positives_num / np.maximum(true_positives_num + false_positives_num,
                                                        np.finfo(np.float64).eps)
            average_precision = self._compute_ap(recall, precision, show_PR_curve=show_PR_curve)
            average_precisions[label] = average_precision, num_annotations
            print(f"{label:>}类\n"
                  f"准确率:{precision[-1]:.4f} "
                  f"虚警率:{1 - precision[-1]:.4f} "
                  f"漏检率:{1 - recall[-1]:.4f} "
                  f"AP值: {average_precision:.4f}")

        present_classes = 0
        precision = 0
        for label, (average_precision, num_annotations) in average_precisions.items():
            if num_annotations > 0:
                present_classes += 1
                precision += average_precision
        mean_ap = precision / present_classes
        print(f"mAP值: {mean_ap:.4f}")
        return mean_ap

    def _compute_ap(self, recall, precision, show_PR_curve):
        """ Compute the average precision, given the recall and precision curves.

        Code originally from https://github.com/rbgirshick/py-faster-rcnn.

        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        if show_PR_curve:
            plt.plot(mrec, mpre, '-*')
        # plt.show()
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        if show_PR_curve:
            plt.plot(mrec, mpre, '-*')
            plt.show()
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def get_real_annotations(self, table):
        res = dict()
        ids = table['ImageID'].values.astype(str)
        labels = table['LabelName'].values.astype(str)
        xmin = table['XMin'].values.astype(np.float32)
        xmax = table['XMax'].values.astype(np.float32)
        ymin = table['YMin'].values.astype(np.float32)
        ymax = table['YMax'].values.astype(np.float32)

        for i in range(len(ids)):
            id = ids[i]
            label = labels[i]
            if id not in res:
                res[id] = dict()
            if label not in res[id]:
                res[id][label] = []
            box = [xmin[i], ymin[i], xmax[i], ymax[i]]
            res[id][label].append(box)

        return res

    def get_detections(self, table):
        res = dict()
        ids = table['ImageID'].values.astype(str)
        labels = table['LabelName'].values.astype(str)
        scores = table['Conf'].values.astype(np.float32)
        xmin = table['XMin'].values.astype(np.float32)
        xmax = table['XMax'].values.astype(np.float32)
        ymin = table['YMin'].values.astype(np.float32)
        ymax = table['YMax'].values.astype(np.float32)

        for i in range(len(ids)):
            id = ids[i]
            label = labels[i]
            if id not in res:
                res[id] = dict()
            if label not in res[id]:
                res[id][label] = []
            box = [xmin[i], ymin[i], xmax[i], ymax[i], scores[i]]
            res[id][label].append(box)

        return res

    def iou(self, boxes, query_boxes):
        """
        Args
            a: (N, 4) ndarray of float
            b: (K, 4) ndarray of float

        Returns
            overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        N = boxes.shape[0]
        K = query_boxes.shape[0]
        iou = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            box_area = (
                    (query_boxes[k, 2] - query_boxes[k, 0]) *
                    (query_boxes[k, 3] - query_boxes[k, 1])
            )
            for n in range(N):
                iw = (
                        min(boxes[n, 2], query_boxes[k, 2]) -
                        max(boxes[n, 0], query_boxes[k, 0])
                )
                if iw > 0:
                    ih = (
                            min(boxes[n, 3], query_boxes[k, 3]) -
                            max(boxes[n, 1], query_boxes[k, 1])
                    )
                    if ih > 0:
                        ua = np.float64(
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) +
                            box_area - iw * ih
                        )
                        iou[n, k] = iw * ih / ua
        return iou


if __name__ == '__main__':
    cal = Cal_mAP().label2csv()
