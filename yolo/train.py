import datetime
import os
import time
import torch.optim as optim
import torch

# from sea.loss import Fusionloss
# from sea.model import FusionNet
from drone_vehicle_dataset import DatasetDroneVehicle
# from sea.utils import RGB2YCrCb, YCrCb2RGB
from yolo.model import YOLOv5s
from torch.cuda.amp import autocast, GradScaler
from yolo.loss import Loss


def train():
    model = YOLOv5s(n_class=5, c1=3, img_size=(512, 640))
    model = model.cuda()

    dataset = DatasetDroneVehicle()
    train_img = dataset.load_img('train', mini=True, only_infrared=True, fused=True)
    # train_img = dataset.load_img('train', mini=True, only_infrared=True)
    train_label = dataset.load_label(mode='train', mini=True, fused=True)
    train_label = dataset.build_labels_for_yolov5(train_label)
    train_img = dataset.split_to_batch(train_img, batch_size=128)
    train_label = dataset.split_to_batch(train_label, batch_size=128)
    n_batch = len(train_img)
    # 定义损失函数和优化器
    criterion = Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.937)

    model.train()

    scaler = GradScaler()

    EPOCH = 200
    # 训练模型
    for epoch in range(EPOCH):
        for i in range(n_batch):
            data = train_img[i]
            label = train_label[i]

            data = data.to('cuda')
            label = [i.to('cuda') for i in label]

            optimizer.zero_grad()

            with autocast():
                outputs = model(data)
                loss, obj_loss, class_loss, bbox_loss = criterion(outputs, label)

            # loss.backward()
            scaler.scale(loss).backward()

            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            if i % 12 == 11:
                print(f'[Epoch {epoch + 1}] [Batch {i + 1:>2}/{n_batch}] Loss:{loss.item():.4f}'
                      f'\t|| obj:{obj_loss:.3f} cls:{class_loss:.3f} box:{bbox_loss:.3f}')
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f'./weight/checkpoint_{epoch}.pth')

    torch.save(model.state_dict(), f'./weight/YOLOv5s_epoch{EPOCH}_FocalLoss_DroneVehicle_Fused.pth')
    # # 评估模型
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    #         100 * correct / total))


if __name__ == '__main__':
    train()
