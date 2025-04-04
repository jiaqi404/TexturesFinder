import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QLineEdit, QWidget, QScrollArea, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from src.utils.transformer_util import fetch_similar, load_model, load_dataset, random_test_image
import os

basedir = os.path.dirname(__file__)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Textures Finder")
        self.setGeometry(100, 100, 1920, 1080)

        # 加载UI
        self.initUI()

        # 设置路径
        print(basedir)
        self.model_path = os.path.join(basedir, "model")
        self.dataset_path = os.path.join(basedir, "dataset")
        self.expansion_path = os.path.join(basedir, "dataset_expansion/original")
        self.processed_path = os.path.join(basedir, "dataset_expansion/processed")
        self.tmp_img_path = os.path.join(basedir, "tmp/test_sample.png")

        # 加载模型与数据集
        self.model, self.extractor, self.device = load_model(self.model_path)
        print("Model loaded")
        self.candidate_subset_path, self.candidate_subset = load_dataset(self.dataset_path, self.expansion_path)
        print("Dataset loaded")

    def update_speed_selection(self):
        sender = self.sender()
        for index, button in enumerate(self.speed_toggle_buttons):
            button.setChecked(button == sender)
            if button == sender:
                self.selected_speed = index
    
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
        self.browse_label = QLabel("1. 选择图片")
        self.browse_label.setFont(bold_font)
        self.left_layout.addWidget(self.browse_label)
        self.browse_button = QPushButton("本地贴图")
        self.browse_button.setFont(normal_font)
        self.browse_button.clicked.connect(self.browse_image)
        self.left_layout.addWidget(self.browse_button)
        self.random_button = QPushButton("随机测试")
        self.random_button.setFont(normal_font)
        self.random_button.clicked.connect(self.random_test_image)
        self.left_layout.addWidget(self.random_button)

        self.param_label = QLabel("2. 设置参数")
        self.param_label.setFont(bold_font)
        self.left_layout.addWidget(self.param_label)

        # 输出图像数量参数
        self.param_layout_1 = QHBoxLayout()
        self.param_label_1 = QLabel("输出图像数量:")
        self.param_label_1.setFont(normal_font)
        self.param_layout_1.addWidget(self.param_label_1)
        self.param_input_1 = QLineEdit()
        self.param_input_1.setPlaceholderText("输入数量参数")
        self.param_layout_1.addWidget(self.param_input_1)
        self.param_input_1.setFont(normal_font)
        self.left_layout.addLayout(self.param_layout_1)

        # 查询速度参数
        self.param_layout_2 = QHBoxLayout()
        self.param_label_2 = QLabel("查询速度及精度:")
        self.param_label_2.setFont(normal_font)
        self.param_layout_2.addWidget(self.param_label_2)

        # 创建三个档位的Toggle按钮
        self.speed_toggle_layout = QHBoxLayout()
        self.speed_toggle_buttons = []
        self.speed_options = ["慢速/高精度", "标准速度/精度", "快速/低精度"]

        for option in self.speed_options:
            button = QPushButton(option)
            button.setCheckable(True)
            button.setFont(normal_font)
            button.clicked.connect(self.update_speed_selection)
            self.speed_toggle_buttons.append(button)
            self.speed_toggle_layout.addWidget(button)

        self.param_layout_2.addLayout(self.speed_toggle_layout)
        self.left_layout.addLayout(self.param_layout_2)

        self.speed_toggle_buttons[1].setChecked(True)
        self.selected_speed = 1

        self.execute_label = QLabel("3. 开始查询")
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
        test_sample = random_test_image(self.processed_path)
        test_sample.save(self.tmp_img_path)
        self.selected_image_path = self.tmp_img_path
        pixmap = QPixmap(self.tmp_img_path)
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
            top_k = int(self.param_input_1.text())
        except ValueError:
            self.image_label.setText("请先填写参数")
            return
        
        try:
            n_components_label = int(self.selected_speed)
        except ValueError:
            self.image_label.setText("请先填写参数")
            return

        similar_images_path = fetch_similar(
            self.model, 
            self.extractor, 
            self.device, 
            self.candidate_subset_path, 
            self.candidate_subset, 
            self.selected_image_path, 
            top_k,
            n_components_label
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