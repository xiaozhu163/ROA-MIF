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
from yolo.test import test_detect
from threading import Thread

from PyQt5.QtCore import pyqtSignal, QObject


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

    def show_result(self):
        file_path = r'D:\python_project\DroneDetect\show\*.jpg'

        img_width = 400
        img_height = img_width // 5 * 4

        pos = ((0, 0), (img_width + 5, 0), (2 * (img_width + 5), 0),
               (0, img_height + 5), (img_width + 5, img_height + 5), ((img_width + 5) * 2, img_height + 5))

        cnt = 0

        for file in glob.glob(file_path):
            self.pixmap = QPixmap(file)
            self.pixmap = self.pixmap.scaled(img_width, img_height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.pixmap = QGraphicsPixmapItem(self.pixmap)
            self.pixmap.setPos(*pos[cnt])
            cnt += 1

            # self.clear()
            self.addItem(self.pixmap)

    def mousePressEvent(self, event):
        print(event.scenePos())


class Signals(QObject):
    text_print = pyqtSignal(str)


class MainWindow():
    def __init__(self):
        super().__init__()

        self.ui = uic.loadUi('Qt_ui/test.ui')
        self.ui.setWindowTitle('算法测试')
        self.scene = Scene(parent=self)

        self.imgView = self.ui.imgView
        self.imgView.setScene(self.scene)

        self.ui.btn_start_test.clicked.connect(self.test)
        self.ui.btn_load_fuse.clicked.connect(self.load_fuse)
        self.ui.btn_load_detect.clicked.connect(self.load_detect)

        self.signal = Signals()
        self.signal.text_print.connect(self.printToGui)

    def printToGui(self, text: str):
        if not text.endswith('\n'):
            text += '\n'
        self.ui.txt_msg.insertPlainText(text)
        self.ui.txt_msg.moveCursor(self.ui.txt_msg.textCursor().End)  # 滚动到最下方

    def test(self):

        # self.ui.txt_msg.setText(msg)
        # self.ui.txt_msg.repaint()
        # self.ui.txt_msg.setText('正在测试')
        self.ui.txt_msg.clear()
        # thread = Thread(target=test_detect,
        #                 args=(self.signal.text_print, self.detect_model_path))
        #
        # thread.start()

        self.scene.show_result()

    def load_fuse(self):
        file_name, _ = QFileDialog.getOpenFileName(self.ui, '加载融合算法',
                                                   directory='./weight',
                                                   filter='(*.pth)')

        if file_name:
            print(file_name)
            self.fuse_model_path = file_name
            self.ui.txt_msg.setText('融合算法加载完成')

    def load_detect(self):
        self.detect_model_path = 'D:/python_project/DroneDetect/yolo/weight/YOLOv5s_epoch200_FocalLoss_DroneVehicle_Fused.pth'
        self.ui.txt_msg.setText('检测框架加载完成')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.ui.show()
    sys.exit(app.exec_())
