from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset
import torchvision.transforms as T
import torch
from tqdm.auto import tqdm
import random
from pathlib import Path
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import numpy as np
import PIL.Image as Image

def load_model():
    model_ckpt = "google/vit-base-patch16-224-in21k"
    extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, extractor, device

def load_dataset():
    # 加载数据集
    dataset_path = Path("dataset")
    dataset_images = list(dataset_path.glob("*.png"))
    candidate_subset_path = [str(img_path) for img_path in dataset_images]
    candidate_subset = [Image.open(img_path) for img_path in dataset_images]

    # 加载 dataset_expansion/original 下的图片
    expansion_path = Path("dataset_expansion/original")
    expansion_images = list(expansion_path.glob("*.png"))
    for img_path in expansion_images:
        image = Image.open(img_path)
        candidate_subset_path.append(str(img_path))
        candidate_subset.append(image)

    return candidate_subset_path, candidate_subset

def random_test_image():
    processed_path = Path("dataset_expansion/processed")
    processed_images = list(processed_path.glob("*.png"))
    test_sample_path = random.choice(processed_images)
    test_sample = Image.open(test_sample_path)

    return test_sample

def fetch_similar(model, extractor, device, candidate_subset_path, candidate_subset, test_img_path, top_k, n_components_label):
    # 加载测试图像
    test_sample = Image.open(test_img_path)

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
        # 如果图像不是 RGB 模式，将其转换为 RGB 模式
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 应用图像预处理链并将其转换为张量
        image = transformation_chain(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(image).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    # 提取候选集中的图像特征
    candidate_embeddings = []
    for image in tqdm(candidate_subset, desc="Extracting candidate embeddings"):
        embedding = extract_embedding(image)
        candidate_embeddings.append(embedding)

    # 使用随机投影降维以提高效率
    candidate_embeddings = np.stack(candidate_embeddings)
    if n_components_label == 0:
        n_components = 384
    elif n_components_label == 1:
        n_components = 256
    elif n_components_label == 2:
        n_components = 128
    else:
        raise ValueError("Invalid n_components_label. Must be 0, 1, or 2.")
    projector = SparseRandomProjection(n_components=n_components, random_state=42)
    reduced_candidate_embeddings = projector.fit_transform(candidate_embeddings)

    test_embedding = extract_embedding(test_sample)
    reduced_test_embedding = projector.transform(test_embedding.reshape(1, -1))

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

    # 返回最相似的图像路径
    top_k_images_path = [candidate_subset_path[idx] for idx in indices[0]]

    return top_k_images_path