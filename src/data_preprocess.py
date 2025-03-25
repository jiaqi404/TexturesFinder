from datasets import load_dataset
import os
import random
from PIL import ImageFilter
from PIL import ImageEnhance

# 加载数据集
dataset = load_dataset("dream-textures/textures-color-normal-1k")

# 创建保存图片的文件夹
original_dir = "dataset_expansion/original"
os.makedirs(original_dir, exist_ok=True)
processed_dir = "dataset_expansion/processed"
os.makedirs(processed_dir, exist_ok=True)

# 抽取前20张图片
num_samples = 20
samples = dataset["train"].select(range(num_samples))["color"]

# 随机添加高斯模糊
def random_gaussian_blur(image, min_raduis=2.0, max_radius=7.0):
    radius = random.uniform(min_raduis, max_radius)
    return image.filter(ImageFilter.GaussianBlur(radius))

# 随机改变色彩平衡
def random_color_balance(image, min_factor=2.0, max_factor=7.0):
    enhancer = ImageEnhance.Color(image)
    factor = random.uniform(min_factor, max_factor)
    return enhancer.enhance(factor)

# 随机旋转图像
def random_rotate(image):
    angles = [90, -90, 180]
    angle = random.choice(angles)
    return image.rotate(angle, expand=True)

# 保存图片到文件夹
for i, sample in enumerate(samples):
    image = sample
    original_path = os.path.join(original_dir, f"img_{i + 1}.png")
    image.save(original_path)

    # 按照次序进行图像处理
    processing_methods = [random_gaussian_blur, random_color_balance, random_rotate]
    for j, processing_method in enumerate(processing_methods):
        processed_image = processing_method(image)
        processed_path = os.path.join(processed_dir, f"img_{i + 1}_processed_{j}.png")
        processed_image.save(processed_path)



