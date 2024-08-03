from PyQt5.QtWidgets import QApplication, QTabWidget, QVBoxLayout, QWidget
import sys
from PyQt5 import uic
from main import MainWindow as MainWindow1
from train import MainWindow as MainWindow2
from test import MainWindow as MainWindow3

class CombinedWindow(QWidget):  # More general class for your application
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        self.tabs = QTabWidget()

        self.window1 = MainWindow1()
        self.window2 = MainWindow2()
        self.window3 = MainWindow3()

        self.tabs.addTab(self.window1.ui, '图像预处理')
        self.tabs.addTab(self.window2.ui, '训练')
        self.tabs.addTab(self.window3.ui, '测试')

        self.layout.addWidget(self.tabs)

        self.setLayout(self.layout)

        self.resize(1600, 1000)  # Set the window size. You can adjust as per your needs

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CombinedWindow()
    window.show()
    sys.exit(app.exec_())