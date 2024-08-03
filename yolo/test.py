import pandas as pd
import torch

from yolo.loss import Loss
from yolo.model import YOLOv5s
from drone_vehicle_dataset import DatasetDroneVehicle
from yolo.visualize import Visualize
from yolo.mAP import mAP_Calculator
import matplotlib.pyplot as plt


def test_detect(ui=None, model_path=None):
    # print(model)
    # print(type(model))
    if ui is not None:
        ui.emit('正在测试...')
    if model_path is None:
        # model = YOLOv5s(n_class=5, c1=1, img_size=(512, 640))
        model = YOLOv5s(n_class=5, c1=3, img_size=(512, 640))
        # model.load_state_dict(torch.load('./weight/YOLOv5s_epoch200_FocalLoss_DroneVehicle.pth'))
        model.load_state_dict(
            torch.load('D:/python_project/DroneDetect/yolo/weight/YOLOv5s_epoch200_FocalLoss_DroneVehicle_Fused.pth'))
        # model.load_state_dict(torch.load('./weight/checkpoint_150.pth'))
        model = model.cuda()
    else:
        model = YOLOv5s(n_class=5, c1=3, img_size=(512, 640))
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
    if ui is not None:
        ui.emit('加载数据...')
    dataset = DatasetDroneVehicle()
    test_img = dataset.load_img('val', mini=False, only_infrared=True, fused=True)
    # test_img = dataset.load_img('val', mini=False, only_infrared=True)
    test_label = dataset.load_label(mode='val', mini=False, fused=True)
    # test_label = dataset.load_label(mode='val', mini=False)
    test_label = dataset.build_labels_for_yolov5(test_label)
    test_img = dataset.split_to_batch(test_img, batch_size=128)
    test_label = dataset.split_to_batch(test_label, batch_size=128)
    n_batch = len(test_img)

    visual = Visualize()

    # model.load_state_dict(torch.load('./weight/YOLOv5s_epoch500_FocalLoss_LEVIR.pth'))
    model = model.to('cuda')

    cls_name = {1: "car", 2: "truck", 3: "bus", 4: 'van', 5: 'freight car'}

    criterion = Loss()

    model.eval()
    img_index = 0
    data_list = []
    with torch.no_grad():
        for i in range(n_batch):
            data = test_img[i]
            label = test_label[i]

            data = data.to('cuda')
            label = [i.to('cuda') for i in label]

            outputs = model(data)

            loss, obj_loss, class_loss, bbox_loss = criterion(outputs, label)
            print(f'[Batch {i + 1}/{n_batch}] Loss:{loss.item():.4f}'
                  f'|||obj:{obj_loss:.3f} cls:{class_loss:.3f} box:{bbox_loss:.3f}')

            outputs = [i.cpu() for i in outputs]
            # break
            ui.emit(f'[Batch {i + 1}/{n_batch}]')

            for index in range(outputs[0].shape[0]):
                pred = outputs[0][index], outputs[1][index], outputs[2][index]
                res = visual.feature_map2bbox(pred, is_pred=True)
                for i in res:
                    conf, cls, x1, y1, w, h = i
                    cls = cls_name[int(cls) + 1]

                    data_list.append(
                        (dataset.val_img_files[img_index].split('\\')[-1], cls, conf, x1, x1 + w, y1, y1 + h))

                img_index += 1
    # 展示图片
    # for index in range(3):
    # index = 10
    # pred = [i.cpu() for i in outputs]
    # preds = [pred[0][index], pred[1][index], pred[2][index]]
    # l = [i.cpu() for i in label]
    # l = [l[0][index], l[1][index], l[2][index]]
    # visual.visual_pred(data[index].cpu(), preds)
    # visual.visual_label(data[index].cpu(), l)
    # plt.show()

    preds = pd.DataFrame(data_list, columns=['ImageID', 'LabelName', 'Conf', 'XMin', 'XMax', 'YMin', 'YMax'])
    # print(preds[:5])

    cal_map = mAP_Calculator()
    cal_map.cal_mAP(preds, show_PR_curve=False)

    if ui is not None:
        ui.emit('测试完成')


if __name__ == '__main__':
    test_detect()
    '''
car类
准确率:0.6660 虚警率:0.3340 漏检率:0.0509 AP值: 0.8961
freight car类
准确率:0.3425 虚警率:0.6575 漏检率:0.7288 AP值: 0.1493
bus类
准确率:0.5278 虚警率:0.4722 漏检率:0.3739 AP值: 0.4722
truck类
准确率:0.2512 虚警率:0.7488 漏检率:0.7034 AP值: 0.1085
van类
准确率:0.4404 虚警率:0.5596 漏检率:0.7655 AP值: 0.1498
mAP值: 0.3552
'''
