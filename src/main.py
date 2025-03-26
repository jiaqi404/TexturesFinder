import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QLineEdit, QWidget, QScrollArea, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from utils.transformer_util import fetch_similar, load_model, load_dataset, random_test_image
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Textures Finder")
        self.setGeometry(100, 100, 1920, 1080)

        # 加载UI
        self.initUI()

        # 加载模型与数据集
        self.model, self.extractor, self.device = load_model()
        print("Model loaded")
        self.candidate_subset_path, self.candidate_subset = load_dataset()
        print("Dataset loaded")
    
    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 设置字体样式
        bold_font = self.font()
        bold_font.setPointSize(12)
        bold_font.setBold(True)
        self.setFont(bold_font)

        normal_font = self.font()
        normal_font.setPointSize(10)
        normal_font.setBold(False)

        # 输入部分
        self.horizontal_layout = QHBoxLayout()

        # 输入部分左侧
        self.left_layout = QVBoxLayout()
        self.browse_label = QLabel("1. 从本地选择一张贴图，或随机一张测试贴图")
        self.browse_label.setFont(bold_font)
        self.left_layout.addWidget(self.browse_label)
        self.browse_button = QPushButton("选择贴图")
        self.browse_button.setFont(normal_font)
        self.browse_button.clicked.connect(self.browse_image)
        self.left_layout.addWidget(self.browse_button)
        self.random_button = QPushButton("随机测试")
        self.random_button.setFont(normal_font)
        self.random_button.clicked.connect(self.random_test_image)
        self.left_layout.addWidget(self.random_button)

        self.param_label = QLabel("2. 选择输出相似贴图的数量")
        self.param_label.setFont(bold_font)
        self.left_layout.addWidget(self.param_label)
        self.param_input = QLineEdit()
        self.param_input.setPlaceholderText("输入参数")
        self.param_input.setFont(normal_font)
        self.left_layout.addWidget(self.param_input)

        self.execute_label = QLabel("3. 点击按钮，输出相似贴图")
        self.execute_label.setFont(bold_font)
        self.left_layout.addWidget(self.execute_label)
        self.execute_button = QPushButton("查找相似贴图")
        self.execute_button.setFont(normal_font)
        self.execute_button.clicked.connect(self.find_similar_images)
        self.left_layout.addWidget(self.execute_button)

        self.left_widget = QWidget()
        self.left_widget.setLayout(self.left_layout)
        self.left_widget.setFixedWidth(580)
        self.horizontal_layout.addWidget(self.left_widget)

        # 输入部分右侧
        self.image_label = QLabel("请先选择一张图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(1200, 480)
        self.horizontal_layout.addWidget(self.image_label, stretch=1)

        # 输出部分
        self.layout.addLayout(self.horizontal_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area_widget = QWidget()
        self.scroll_area_layout = QHBoxLayout(self.scroll_area_widget)
        self.scroll_area_layout.setAlignment(Qt.AlignLeft)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.layout.addWidget(self.scroll_area)

        self.selected_image_path = None

    def random_test_image(self):
        test_sample = random_test_image()
        test_sample.save("src/tmp/test_sample.png")
        self.selected_image_path = "src/tmp/test_sample.png"
        pixmap = QPixmap("src/tmp/test_sample.png")
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def browse_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_path:
            self.selected_image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def open_image_path(self, path):
        os.startfile(path)

    def find_similar_images(self):
        if not self.selected_image_path:
            self.image_label.setText("请先选择一张贴图")
            return

        try:
            param = int(self.param_input.text())
        except ValueError:
            self.image_label.setText("请先输出参数")
            return

        similar_images_path = fetch_similar(
            self.model, 
            self.extractor, 
            self.device, 
            self.candidate_subset_path, 
            self.candidate_subset, 
            self.selected_image_path, 
            param
        )

        # 清空过往的相似贴图
        for i in reversed(range(self.scroll_area_layout.count())):
            widget = self.scroll_area_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # 显示相似贴图
        for image_path in similar_images_path:
            pixmap = QPixmap(image_path)
            label = QLabel()
            label.setPixmap(pixmap.scaled(480, 480, Qt.KeepAspectRatio))
            label.setCursor(Qt.PointingHandCursor)
            label.mousePressEvent = lambda event, path=image_path: self.open_image_path(path)
            self.scroll_area_layout.addWidget(label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())