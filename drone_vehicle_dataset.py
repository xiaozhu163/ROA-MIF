import glob
import os
import random
import shutil
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from typing import Literal
import cv2
import torch.nn.functional as F


class DatasetDroneVehicle:
    n_class = 5

    def __init__(self, ui=None, img_size=640):
        self.path = r'D:\dataset\DroneVehicle\processed/'
        self.ui = ui
        self.img_size = (int(img_size) // 5 * 4, int(img_size))
        self.cuda = torch.cuda.is_available()
        # vis表示可见光图片，ir表示红外图片
        self.train_img_files, self.train_imgr_files, self.train_label_files, self.train_labelr_files = \
            self.get_files(mode='train')
        self.val_img_files, self.val_imgr_files, self.val_label_files, self.val_labelr_files = \
            self.get_files(mode='val')
        self.test_img_files, self.test_imgr_files, self.test_label_files, self.test_labelr_files = \
            self.get_files(mode='test')

        mini_size = 3000

        self.train_img_files_mini = self.train_img_files[:mini_size]
        self.train_imgr_files_mini = self.train_imgr_files[:mini_size]
        self.train_label_files_mini = self.train_label_files[:mini_size]
        self.train_labelr_files_mini = self.train_labelr_files[:mini_size]

    def get_files(self, mode: Literal['train', 'test', 'val']):
        imgs_path = self.path + mode + '/img/'
        imgrs_path = self.path + mode + '/imgr/'

        img_files = glob.glob(imgs_path + '*.jpg')
        imgr_files = glob.glob(imgrs_path + '*.jpg')

        label_path = self.path + mode + '/label/'
        labelr_path = self.path + mode + '/labelr/'
        label_files = glob.glob(label_path + '*.txt')
        labelr_files = glob.glob(labelr_path + '*.txt')

        if mode == 'train':
            random.seed(0)
            random.shuffle(img_files)
            random.seed(0)
            random.shuffle(imgr_files)

            random.seed(0)
            random.shuffle(label_files)
            random.seed(0)
            random.shuffle(labelr_files)

        return img_files, imgr_files, label_files, labelr_files

    def load_cache(self, mode='train', mini=None):
        self._print('正在读取缓存...')

        assert mode in ('train', 'val', 'test'), '参数"mode"必须为train, test, val中一项'

        if mini is None:
            if mode == 'train':
                mini = True
            elif mode in ('test', 'val'):
                mini = False

        mini = '_mini' if mini else ''

        file = f'D:/python_project/DroneDetect/cache/DroneVehicle_{mode}_img' + mini + '.pt'
        img = torch.load(file)
        self._print('读取缓存完毕。')

        return img

    def load_img(self, mode='train', mini=None, only_infrared=False, fused=False):
        self._print('正在读取图像...')

        assert mode in ('train', 'val', 'test'), '参数"mode"必须为train, test, val中一项'

        if mini is None:
            if mode == 'train':
                mini = True
            elif mode in ('test', 'val'):
                mini = False

        if self.img_size == (512, 640):
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor()
            ])

        if mode == 'train':
            if mini:
                files = self.train_img_files_mini
                if fused:
                    files.extend(self.val_img_files[:700])
                rfiles = self.train_imgr_files_mini
            else:
                files = self.train_img_files
                rfiles = self.train_imgr_files
        elif mode == 'val':
            files = self.val_img_files
            rfiles = self.val_imgr_files
        else:
            files = self.test_img_files
            rfiles = self.test_imgr_files

        img_list = []

        for i in tqdm(range(len(files))):
            if fused:
                if mode == 'train' or mode == 'val':
                    file_name = files[i].replace(f'{mode}/img\\', f'{mode}/fuse_img\\')
                    file_name = file_name.replace(f'val/img\\', f'val/fuse_img\\')
                    img = Image.open(file_name)
                    img = transform(img)
                    img_list.append(img)
                    continue
                else:
                    raise NotImplementedError

            imgr = Image.open(rfiles[i]).convert('L')
            imgr = transform(imgr)

            if not only_infrared:
                img = Image.open(files[i])
                img = transform(img)
                imgr = torch.cat((img, imgr), dim=0)

            img_list.append(imgr)

        img_list = torch.stack(img_list)
        self._print('读取数据完成')
        return img_list

    def load_label(self, mode='train', mini=None, fused=False):
        self._print('正在读取标签...')

        assert mode in ('train', 'val', 'test'), '参数"mode"必须为train, test, val中一项'

        if mini is None:
            if mode == 'train':
                mini = True
            elif mode in ('test', 'val'):
                mini = False

        if mode == 'train':
            if mini:
                # files = self.train_label_files_mini
                rfiles = self.train_labelr_files_mini
                if fused:
                    rfiles.extend(self.val_labelr_files[:700])
            else:
                # files = self.train_label_files
                rfiles = self.train_labelr_files
        elif mode == 'val':
            # files = self.val_label_files
            rfiles = self.val_labelr_files
        else:
            # files = self.test_label_files
            rfiles = self.test_labelr_files

        import warnings
        warnings.filterwarnings("ignore")

        label_list = []
        for i in tqdm(range(len(rfiles))):
            labelr = np.loadtxt(rfiles[i], delimiter=' ').reshape((-1, 5))
            if labelr.size > 0:
                labelr[:, [3, 4]] = labelr[:, [3, 4]] - labelr[:, [1, 2]]
                labelr[:, [1, 2]] = labelr[:, [1, 2]] + labelr[:, [3, 4]] / 2

            label_list.append(labelr)

        self._print('读取标签完成')
        return label_list

    def build_labels_for_yolov5(self, label_list):
        img_width = 640
        img_height = 512
        feature_size = [(64, 80), (32, 40), (16, 20)]  # 图片宽、高总共有多少个gird
        grid_pixel = (8, 16, 32)  # 每个grid宽高各占多少像素
        n_detect_feature = 3  # 有几个尺度的特征图
        anchors = ([(24, 34), (46, 21), (27, 57)], [(61, 28), (56, 40), (39, 63)], [(62, 58), (49, 131), (105, 77)])
        anchors_np = np.array(anchors)  # 转换为numpy.array格式的anchors
        n_class = 5  # 目标类别数量
        n_anchor = 3  # 单个特征图下，每个grid中anchor数量
        ratio_threshold = 3.0  # anchor的宽与gt的bbox宽之比要介于1/3与3.0之间（两者高之比也满足此条件）时，这个anchor负责检测这个gt

        label_final = [[], [], []]
        for label in tqdm(label_list):

            label = torch.from_numpy(label)

            for i in range(n_detect_feature):  # 对每个特征图操作
                label_tensor = torch.zeros(feature_size[i][0], feature_size[i][1], n_anchor, 5 + n_class,
                                           dtype=torch.float64)
                if label.size == 0:
                    label_final[i].append(label_tensor)
                    continue
                ratio = label[:, None, [3, 4]] / anchors_np[None, i]  # 每个目标的bbox宽高与每个anchor宽高之比
                ratio = torch.maximum(ratio, 1 / ratio)  # 把比值小于1的取倒数
                ratio = torch.max(ratio, dim=2)[0]  # 取宽之比、高之比中更悬殊的，如果更悬殊的也在threshold内，那么就是正样本
                index = ratio < ratio_threshold
                index = torch.vstack((index, index, index))  # 因为gird旁边的两个grid也作为正样本，所以index要变成3倍

                # 中心在第几行第几列的grid
                grid_yx = torch.div(label[:, [2, 1]], grid_pixel[i], rounding_mode='floor').long()
                # 中心在grid中的位置
                center_xy = label[:, [1, 2]] % grid_pixel[i] / grid_pixel[i]

                # 根据中心位置，将旁边的两个grid也加入到grid_yx中
                # 可能在同一个grid中包含两个正样本，但是前一个会被覆盖掉，所以没关系
                extra_x = torch.where(center_xy[:, [0]] >= 0.5, 1, -1)
                extra_x = torch.hstack((torch.zeros(extra_x.shape), extra_x))
                extra_y = torch.where(center_xy[:, [1]] >= 0.5, 1, -1)
                extra_y = torch.hstack((extra_y, torch.zeros(extra_y.shape)))
                extra_yx = torch.vstack((extra_x + grid_yx, extra_y + grid_yx))
                # 越界判断
                extra_yx = torch.maximum(extra_yx, torch.zeros(extra_yx.shape))
                extra_yx = torch.minimum(extra_yx, torch.tensor(feature_size[i]) - 1)

                grid_yx = torch.vstack((grid_yx, extra_yx)).long()

                # 构建label中的xywh
                label_wh = label[:, [3, 4]] / grid_pixel[i]
                label_wh = torch.vstack((label_wh, label_wh, label_wh))
                if 0 in label_wh:
                    exit()
                label_xy = torch.vstack((center_xy,
                                         center_xy + extra_x[:, [1, 0]] * -1,
                                         center_xy + extra_y[:, [1, 0]] * -1))
                label_xywh = torch.hstack((label_xy, label_wh))
                # 构建label中的confidence
                label_confidence = torch.ones((label_xywh.shape[0], 1))
                # 构建label中的one-hot
                label_one_hot_class = F.one_hot(label[:, 0].long(), num_classes=n_class)
                label_one_hot_class = torch.vstack((label_one_hot_class, label_one_hot_class, label_one_hot_class))
                label_all = torch.hstack((label_confidence, label_one_hot_class, label_xywh))
                # 对应grid分配label
                label_tensor[grid_yx[:, 0], grid_yx[:, 1]] = label_all[:, None]
                temp = label_tensor[grid_yx[:, 0], grid_yx[:, 1]]
                temp[~index] = 0
                label_tensor[grid_yx[:, 0], grid_yx[:, 1]] = temp

                # 3*8变24
                label_tensor = label_tensor.view(label_tensor.shape[0], label_tensor.shape[1], -1)
                label_final[i].append(label_tensor)

        label_final = [torch.stack(i) for i in label_final]
        return label_final

    def split_to_batch(self, img, batch_size=128):
        print('spliting...')
        if isinstance(img, list):
            n_sample = img[0].shape[0]
            # 需要分成多少块
            chunk_size = (n_sample - 1) // batch_size + 1
            img = [torch.chunk(i, chunk_size, dim=0) for i in img]
            img = list(zip(*img))
            # imgr = torch.chunk(imgr, chunk_size, dim=0)

            # data = list(zip(img, imgr))
            return img

        n_sample = img.shape[0]
        # 需要分成多少块
        chunk_size = (n_sample - 1) // batch_size + 1
        img = torch.chunk(img, chunk_size, dim=0)
        # imgr = torch.chunk(imgr, chunk_size, dim=0)

        # data = list(zip(img, imgr))
        return img

    def cache_img(self, mode: Literal['train', 'test', 'val']):
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])
        if mode == 'train':
            files = self.train_img_files_mini
            rfiles = self.train_imgr_files_mini
        elif mode == 'val':
            files = self.val_img_files
            rfiles = self.val_imgr_files
        else:
            files = ''
            rfiles = ''

        img_list = []

        for i in tqdm(range(len(files))):
            img = Image.open(files[i])
            imgr = Image.open(rfiles[i]).convert('L')
            img = transform(img)
            imgr = transform(imgr)

            img = torch.cat((img, imgr), dim=0)
            img_list.append(img)
        img_list = torch.stack(img_list)
        self._print('读取数据完成')
        # self._print("均值：", torch.mean(img_list, dim=(0, 2, 3)))
        # self._print("标准差：", torch.std(img_list, dim=(0, 2, 3)))
        torch.save(img_list, f'./cache/DroneVehicle_{mode}_img_mini.pt')

    def kmean_anchors(self, n=9, img_size=640, thr=3.0, gen=10000, verbose=False, wh=None):
        """ Creates kmeans-evolved anchors from training dataset

            Arguments:
                dataset: path to data.yaml, or a loaded dataset
                n: number of anchors
                img_size: image size used for training
                thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
                gen: generations to evolve anchors using genetic algorithm
                verbose: print all results

            Return:
                k: kmeans evolved anchors

            Usage:
                from utils.autoanchor import *; _ = kmean_anchors()
        """
        from scipy.cluster.vq import kmeans

        npr = np.random
        thr = 1 / thr

        def metric(k, wh):  # compute metrics
            r = wh[:, None] / k[None]
            x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
            # x = wh_iou(wh, torch.tensor(k))  # iou metric
            return x, x.max(1)[0]  # x, best_x

        def anchor_fitness(k):  # mutation fitness
            _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
            return (best * (best > thr).float()).mean()  # fitness

        def print_results(k, verbose=True):
            k = k[np.argsort(k.prod(1))]  # sort small to large
            x, best = metric(k, wh0)
            bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
            s = f'thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
                f'n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
                f'past_thr={x[x > thr].mean():.3f}-mean: '
            for x in k:
                s += '%i,%i, ' % (round(x[0]), round(x[1]))
            if verbose:
                print(s[:-2])
            return k

        # Get label wh
        # wh0 = np.load('./cache/train_object_bbox.npy')
        wh0 = wh

        # Filter
        i = (wh0 < 3.0).any(1).sum()
        if i:
            print(f'WARNING: Extremely small objects found: {i} of {len(wh0)} labels are < 3 pixels in size')
        wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels
        # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

        # Kmeans init
        try:
            assert n <= len(wh)  # apply overdetermined constraint
            s = wh.std(0)  # sigmas for whitening
            k = kmeans(wh / s, n, iter=30)[0] * s  # points
            assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
        except Exception:
            print(f'WARNING: switching strategies from kmeans to random init')
            k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
        wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
        k = print_results(k, verbose=False)

        # Plot
        # k, d = [None] * 20, [None] * 20
        # for i in tqdm(range(1, 21)):
        #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
        # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
        # ax = ax.ravel()
        # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
        # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
        # ax[0].hist(wh[wh[:, 0]<100, 0],400)
        # ax[1].hist(wh[wh[:, 1]<100, 1],400)
        # fig.savefig('wh.png', dpi=200)

        # Evolve
        f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
        pbar = tqdm(range(gen), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        for _ in pbar:
            v = np.ones(sh)
            while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
                v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
            kg = (k.copy() * v).clip(min=2.0)
            fg = anchor_fitness(kg)
            if fg > f:
                f, k = fg, kg.copy()
                pbar.desc = f'Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
                if verbose:
                    print_results(k, verbose)

        return print_results(k)

    def _print(self, s):
        if self.ui is None:
            print(s)
        else:
            self.ui.emit(s + '\n')
            # self.ui.repaint()


if __name__ == '__main__':
    dataset = DatasetDroneVehicle()
    img = dataset.load_img('train', mini=True)
    # l = dataset.build_labels_for_yolov5(label_list)
    # for i in l:
    #     print(i.shape)
    print(img.shape)
    # print(dataset.img_files)
    # dataset.cache_img('val')
    # img = dataset.load_img('val')
    # print(img.shape)
    # train_dataset = DatasetDroneVehicle()
    # train_img = train_dataset.load_img(mode='val')
    # train_data = train_dataset.split_to_batch(train_img, batch_size=32)
    # d = train_data[0].cuda()
    # print(d.shape)
    # print(train_data[0].shape)
    # print("the training dataset is length:{}".format(len(train_data)))
