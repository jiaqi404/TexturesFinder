from transformers import AutoFeatureExtractor, AutoModel
import torchvision.transforms as T
import torch
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import PIL.Image as Image
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

# 加载图像编码器
model_ckpt = "google/vit-base-patch16-224-in21k"
model_dir = Path("model")
model_dir.mkdir(exist_ok=True)

# 检查模型是否已保存
if not (model_dir / "config.json").exists():
    extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)
    extractor.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
else:
    extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)

hidden_dim = model.config.hidden_size
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 加载数据集
dataset_path = Path("dataset")
dataset_images = list(dataset_path.glob("*.png"))
candidate_subset = [Image.open(img_path) for img_path in dataset_images]

# 加载 dataset_expansion/original 下的图片
expansion_path = Path("dataset_expansion/original")
expansion_images = list(expansion_path.glob("*.png"))
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

# 提取测试图像的特征
test_embedding = extract_embedding(test_sample)

# 提取候选集中的图像特征
candidate_embeddings = []
for image in tqdm(candidate_subset, desc="Extracting candidate embeddings"):
    embedding = extract_embedding(image)
    candidate_embeddings.append(embedding)

# 使用随机投影降维以提高效率
candidate_embeddings = np.stack(candidate_embeddings)
projector = SparseRandomProjection(n_components=256, random_state=42)
reduced_candidate_embeddings = projector.fit_transform(candidate_embeddings)

test_embedding = extract_embedding(test_sample)
reduced_test_embedding = projector.transform(test_embedding.reshape(1, -1))

top_k = 5

# 使用BallTree计算最相似图像
normalized_candidates = normalize(reduced_candidate_embeddings, norm="l2")
normalized_test = normalize(reduced_test_embedding, norm="l2")

nn = NearestNeighbors(
    n_neighbors=top_k,
    metric="euclidean",
    algorithm="ball_tree",
    n_jobs=-1
)
nn.fit(normalized_candidates)
distances, indices = nn.kneighbors(normalized_test)

fig, axes = plt.subplots(1, top_k + 1, figsize=(20, 5))
axes[0].imshow(np.array(test_sample))
axes[0].axis("off")
axes[0].set_title("Test Image")

for i, idx in enumerate(indices[0]):
    axes[i + 1].imshow(np.array(candidate_subset[idx]))
    axes[i + 1].axis("off")
    axes[i + 1].set_title(f"Similar {i+1}")

plt.tight_layout()
plt.savefig("similar_images.png")
print("Similar images saved as similar_images.png!")