from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset
import torchvision.transforms as T
import torch
from tqdm.auto import tqdm
import random
from pathlib import Path
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

def fetch_similar(model, extractor, device, candidate_subset_path, candidate_subset, test_img_path, top_k):
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

    # 返回最相似的图像
    top_k_indices = torch.topk(cosine_similarities, top_k).indices
    top_k_images_path = [candidate_subset_path[idx] for idx in top_k_indices]

    return top_k_images_path