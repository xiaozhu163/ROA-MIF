import glob
import sys
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsRectItem, QGraphicsScene, QGraphicsPixmapItem, \
    QFileDialog, QAction, QGraphicsTextItem
from PyQt5.QtGui import QPixmap, QImage, QPen, QCursor, QTransform, QFont
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, QRectF, QRect
from PyQt5.QtWidgets import QMainWindow, QMenu
from PyQt5 import uic
from pathlib import Path
import shutil
import time
import numpy as np
import cv2
import random


class Scene(QGraphicsScene):
    def __init__(self, parent):
        super().__init__()

        self.img_index = None
        self.mode = None
        self.parent = parent

        # self.draw = False
        self.origin = None
        self.current_rect = None
        self.angle = 0  # 图片旋转的角度
        self.label = []

        self.rect = QGraphicsRectItem()
        self.rect.setRect(0, 0, 500, 500)
        self.rect.setPen(QPen(Qt.red, 5))

    def open_img(self, mode):
        file_name, _ = QFileDialog.getOpenFileName(self.parent.ui, '打开图片',
                                                   directory=r'D:\dataset\LEVIR\LEVIR\imageWithLabel',
                                                   filter='Image Files (*.png *.jpg *.bmp)')

        if file_name:
            self.mode = mode
            file_name = file_name.replace('/', '\\')
            self.img_dirs = Path(file_name).parent
            self.img_list = glob.glob(str(self.img_dirs) + '/*.jpg')
            self.img_index = self.img_list.index(file_name)
            self.img_list_len = len(self.img_list)

            self._show_img(file_name)

    def next_img(self):
        if self.img_index is None:
            return
        if self.img_index == len(self.img_list) - 1:
            pass
        else:
            self.img_index += 1
            self._show_img(self.img_list[self.img_index])
            self.angle = 0
            self.parent.ui.txt_msg.clear()
            self.label = []

    def last_img(self):
        if self.img_index is None:
            return

        if self.img_index == 0:
            pass
        else:
            self.img_index -= 1
            self._show_img(self.img_list[self.img_index])
            self.angle = 0
            self.label = []

    def _show_img(self, file_name):
        self.pixmap = QPixmap(file_name)
        self.clear()
        self.addPixmap(self.pixmap)
        self.parent.ui.txt_img_number.setPlainText(f"{self.img_index + 1} / {self.img_list_len}")

    def rotate(self):
        if self.img_index is None:
            return
        angle = float(self.parent.ui.edt_angle.text())
        if angle == 0:
            return
        self.angle += angle
        pixmap = QPixmap(self.img_list[self.img_index])
        self.pixmap = pixmap.transformed(QTransform().rotate(self.angle))
        self.clear()
        self.addPixmap(self.pixmap)

    def gamma(self):
        if self.img_index is None:
            return
        file_name = self.img_list[self.img_index]
        img = cv2.imread(file_name)

        # 直方图均衡化
        # r, g, b = cv2.split(img)
        # r1 = cv2.equalizeHist(r)
        # g1 = cv2.equalizeHist(g)
        # b1 = cv2.equalizeHist(b)
        # image_lap = cv2.merge([r1, g1, b1])

        # gamma变换
        fgamma = 2
        image_gamma = np.uint8(np.power((np.array(img) / 255.0), fgamma) * 255.0)
        cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
        cv2.convertScaleAbs(image_gamma, image_gamma)

        # 拉普拉斯变换
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # image_lap = cv2.filter2D(img, cv2.CV_8UC3, kernel)

        # cv2 to pixmap
        height, width, channel = image_gamma.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_gamma.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # 将QImage转换为QPixmap
        self.pixmap = QPixmap.fromImage(q_image)
        self.clear()
        self.addPixmap(self.pixmap)
        # return image_lap

    def save_all_img(self):
        if self.mode is None:  # 未加载图像，则不保存，直接返回
            return

        save_path = self.parent.save_img_path + self.mode + '/'
        cnt = 0
        for img_file in self.img_list:
            if cnt % 500 == 0:
                self.parent.ui.txt_msg.setPlainText(f'已保存{cnt}/{self.img_list_len}')
                self.parent.ui.txt_msg.repaint()
            shutil.copy(img_file, save_path)
            cnt += 1
        self.parent.ui.txt_msg.setPlainText('保存完成')

    def save_img(self):
        # 保存预处理后的图片
        if self.img_index is None:
            return
        file_name = self.img_list[self.img_index].split('\\')[-1].split('.')[0]
        save_path = f'./data/img_processed/{self.mode}/{file_name}_2.jpg'
        self.pixmap.save(save_path)
        self.parent.ui.txt_msg.setPlainText('已保存')

    def save_label(self):
        if self.img_index is None:
            return
        file_name = self.img_list[self.img_index].split('\\')[-1]
        file_name = f'./data/img/{self.mode}/{file_name}'
        # save_path = f'./data/label/{self.mode}/{file_name}_2.jpg'
        with open('./data/label/label.txt', 'a+') as w:
            for label in self.label:
                # label为cls,x1,y1,x2,y2。self.label为label的列表
                label = [str(i) for i in label]
                label = ' '.join(label)
                # print(self.label)
                s = f"{file_name} {label}\n"
                w.write(s)

    def get_train_test_data(self):
        train_rate = 0.8
        random.seed(0)

        file_name = glob.glob('./data/img_processed/*/*.jpg')
        random.shuffle(file_name)

        n_train = int(len(file_name) * train_rate)
        train_file_name = file_name[:n_train]
        test_file_name = file_name[n_train:]

        for file in train_file_name:
            shutil.copy(file, './data/train/')
        for file in test_file_name:
            shutil.copy(file, './data/test/')

        self.parent.ui.txt_msg.setPlainText('已生成训练集与测试集')
        # print(file_name)

    def mousePressEvent(self, event):
        if self.img_index is None:
            return
        if event.button() == Qt.LeftButton:
            if self.parent.ui.check_labeling.checkState() or self.parent.ui.check_clipping.checkState():
                self.origin = event.scenePos()
                self.current_rect = QGraphicsRectItem(QRectF(self.origin, self.origin))
                self.current_rect.setPen(QPen(Qt.red, 2))
                self.addItem(self.current_rect)

                self.class_text = QGraphicsTextItem(self.parent.ui.edt_class.text())
                font = QFont()
                font.setPointSize(15)
                self.class_text.setFont(font)
                self.class_text.setPos(self.origin)
                self.class_text.setDefaultTextColor(QColor('red'))
                self.addItem(self.class_text)

    def mouseMoveEvent(self, event):
        if self.img_index is None:
            return
        print(event.scenePos())
        if self.current_rect:
            rect = QRectF(self.origin, event.scenePos()).normalized()
            self.current_rect.setRect(rect)

    def mouseReleaseEvent(self, event):
        """

        """
        if self.img_index is None:
            return
        if event.button() == Qt.LeftButton and \
                (self.parent.ui.check_labeling.checkState() or self.parent.ui.check_clipping.checkState()):
            x2, y2 = int(event.scenePos().x()), int(event.scenePos().y())
            x1, y1 = int(self.origin.x()), int(self.origin.y())
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            self.current_rect = None

            if self.parent.ui.check_labeling.checkState():

                cls = self.parent.ui.edt_class.text()
                if cls == '':
                    cls = '0'

                self.label.append((cls, x1, y1, x2, y2))

            elif self.parent.ui.check_clipping.checkState():
                print(x1, y1, x2, y2)

                self.pixmap = self.pixmap.copy(x1, y1, x2 - x1, y2 - y1)
                self.clear()
                self.addPixmap(self.pixmap)


class MainWindow():
    def __init__(self):
        super().__init__()

        self.ui = uic.loadUi('Qt_ui/preprocess.ui')
        self.ui.setWindowTitle('图片预处理')
        self.scene = Scene(parent=self)

        self.save_img_path = './data/img/'

        self.imgView = self.ui.imgView
        self.imgView.setScene(self.scene)

        self.ui.btn_load_RGB.clicked.connect(lambda: self.scene.open_img(mode="RGB"))
        self.ui.btn_load_SAR.clicked.connect(lambda: self.scene.open_img(mode="SAR"))
        self.ui.btn_load_IR.clicked.connect(lambda: self.scene.open_img(mode="IR"))

        self.ui.btn_save_all_img.clicked.connect(self.scene.save_all_img)
        self.ui.btn_save_img.clicked.connect(self.scene.save_img)
        self.ui.btn_save_label.clicked.connect(self.scene.save_label)
        self.ui.btn_get_train_test_data.clicked.connect(self.scene.get_train_test_data)

        self.ui.btn_next_img.clicked.connect(self.scene.next_img)
        self.ui.btn_last_img.clicked.connect(self.scene.last_img)
        self.ui.btn_rotate.clicked.connect(self.scene.rotate)
        self.ui.btn_gamma.clicked.connect(self.scene.gamma)

        self.ui.txt_img_number.setPlaceholderText('未加载图片')
        # self.ui.txt_angle.setPlainText('度')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.ui.show()
    sys.exit(app.exec_())
