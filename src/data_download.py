from datasets import load_dataset
import os
import random

# 加载数据集
dataset = load_dataset("dream-textures/textures-color-normal-1k")

# 创建保存图片的文件夹
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

# 去掉前20张图片，因为他们被选择进行了预处理
dataset["train"] = dataset["train"].select(range(20, dataset["train"].num_rows))
# 在剩下的中随机200张图片作为候选集
num_samples = 200
seed = random.randint(0, 1000)
dataset_subset = dataset["train"].shuffle(seed=seed).select(range(num_samples))["color"]

# 保存图片到文件夹
for i, image in enumerate(dataset_subset):
    original_path = os.path.join(dataset_dir, f"img_{i + 1}.png")
    image.save(original_path)