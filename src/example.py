from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset
import torchvision.transforms as T
import torch
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import PIL.Image as Image

# 加载图像编码器
model_ckpt = "google/vit-base-patch16-224-in21k"
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
hidden_dim = model.config.hidden_size
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 加载数据集
dataset = load_dataset("dream-textures/textures-color-normal-1k")
# 去掉前20张图片，因为他们被选择进行了预处理
dataset["train"] = dataset["train"].select(range(20, dataset["train"].num_rows))
# 在剩下的中随机200张图片作为候选集
num_samples = 200
seed = random.randint(0, 1000)
candidate_subset = dataset["train"].shuffle(seed=seed).select(range(num_samples))["color"]

# 加载 dataset_expansion/original 下的图片
expansion_path = Path("dataset_expansion/original")
expansion_images = list(expansion_path.glob("*.png"))
# 将图片并入候选集
for img_path in expansion_images:
    image = Image.open(img_path)
    candidate_subset.append(image)

# 随机选择 dataset_expansion/processed 下的一张图片作为测试图像
processed_path = Path("dataset_expansion/processed")
processed_images = list(processed_path.glob("*.png"))
test_sample_path = random.choice(processed_images)
test_sample = Image.open(test_sample_path)

# 图像预处理方法
transformation_chain = T.Compose(
    [
        T.Resize(int((256 / 224) * extractor.size["height"])),
        T.CenterCrop(extractor.size["height"]),
        T.ToTensor(),
        T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
    ]
)

# 提取特征方法
def extract_embedding(image):
    if image.mode != "RGB":
        image = image.convert("RGB")  # Convert grayscale to RGB
    image = transformation_chain(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image).last_hidden_state.mean(dim=1).squeeze().cpu()
    return embedding

# 提取候选集中的图像特征
candidate_embeddings = []
for image in tqdm(candidate_subset, desc="Extracting candidate embeddings"):
    embedding = extract_embedding(image)
    candidate_embeddings.append(embedding)

candidate_embeddings = torch.stack(candidate_embeddings)

# 提取测试图像的特征
test_embedding = extract_embedding(test_sample)

# 计算余弦相似度
cosine_similarities = torch.nn.functional.cosine_similarity(
    test_embedding.unsqueeze(0), candidate_embeddings
)

# 显示测试图像和最相似的 5 张图像
top_k = 5
top_k_indices = torch.topk(cosine_similarities, top_k).indices

fig, axes = plt.subplots(1, top_k + 1, figsize=(20, 5))
axes[0].imshow(np.array(test_sample))
axes[0].axis("off")
axes[0].set_title("Test Image")

for i, idx in enumerate(top_k_indices):
    axes[i + 1].imshow(np.array(candidate_subset[idx]))
    axes[i + 1].axis("off")
    axes[i + 1].set_title(f"Similar {i+1}")

plt.tight_layout()
plt.savefig("outputs/similar_images.png")