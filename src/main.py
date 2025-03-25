import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QLineEdit, QWidget, QScrollArea, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from utils.transformer_util import fetch_similar, load_model, load_dataset

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Textures Finder")
        self.setGeometry(100, 100, 720, 720)

        self.initUI()
        self.model, self.extractor, self.device = load_model()
        print("Model loaded")
        self.candidate_subset = load_dataset()
        print("Dataset loaded")
    
    # UI
    def initUI(self):
        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create a horizontal layout for the left and right sections
        self.horizontal_layout = QHBoxLayout()

        # Left section layout
        self.left_layout = QVBoxLayout()
        self.browse_button = QPushButton("选择图片")
        self.browse_button.clicked.connect(self.browse_image)
        self.left_layout.addWidget(self.browse_button)

        self.param_label = QLabel("输出相似图片数量")
        self.left_layout.addWidget(self.param_label)
        self.param_input = QLineEdit()
        self.param_input.setPlaceholderText("输入参数")
        self.left_layout.addWidget(self.param_input)

        # Wrap the left layout in a QWidget to control its width
        self.left_widget = QWidget()
        self.left_widget.setLayout(self.left_layout)
        self.left_widget.setFixedWidth(320)  # Adjust the width as needed
        self.horizontal_layout.addWidget(self.left_widget)

        # Right section layout
        self.image_label = QLabel("请先选择一张图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(320, 240)
        self.horizontal_layout.addWidget(self.image_label, stretch=1)  # Allow the right section to expand

        # Add the horizontal layout to the main layout
        self.layout.addLayout(self.horizontal_layout)

        # Execute button
        self.execute_button = QPushButton("查找相似贴图")
        self.execute_button.clicked.connect(self.find_similar_images)
        self.layout.addWidget(self.execute_button)

        # Scroll area for displaying results
        self.scroll_area = QScrollArea()
        self.scroll_area_widget = QWidget()
        self.scroll_area_layout = QHBoxLayout(self.scroll_area_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.layout.addWidget(self.scroll_area)

        self.selected_image_path = None

    def browse_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_path:
            self.selected_image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def find_similar_images(self):
        if not self.selected_image_path:
            self.image_label.setText("Please select an image first!")
            return

        try:
            param = int(self.param_input.text())
        except ValueError:
            self.image_label.setText("Please enter a valid integer parameter!")
            return

        # Fetch similar images
        similar_images = fetch_similar(self.model, self.extractor, self.device, self.candidate_subset, self.selected_image_path, param)

        # Clear previous results
        for i in reversed(range(self.scroll_area_layout.count())):
            widget = self.scroll_area_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # Display similar images
        for similar_image in similar_images:
            pil_image = similar_image.convert("RGBA")
            data = pil_image.tobytes("raw", "RGBA")
            q_image = QImage(data, pil_image.size[0], pil_image.size[1], QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(q_image)
            label = QLabel()
            label.setPixmap(pixmap.scaled(200, 150, Qt.KeepAspectRatio))
            self.scroll_area_layout.addWidget(label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())