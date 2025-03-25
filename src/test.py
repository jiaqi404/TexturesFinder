from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset
import torchvision.transforms as T
import torch
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random

# Load image encoder
model_ckpt = "google/vit-base-patch16-224-in21k"
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
hidden_dim = model.config.hidden_size

# Load the dataset
dataset = load_dataset("dream-textures/textures-color-normal-1k")
num_samples = 200
seed = random.randint(0, 1000)
candidate_subset = dataset["train"].shuffle(seed=seed).select(range(num_samples))["color"]

# Display 10 random images from the dataset
fig, axes = plt.subplots(1, 10, figsize=(20, 5))
for i, ax in enumerate(axes):
    sample = candidate_subset[i]
    ax.imshow(np.array(sample))
    ax.axis("off")
    ax.set_title(f"Image {i+1}")
plt.tight_layout()
plt.savefig("dataset_images2.png")

# Test image
test_idx = np.random.choice(dataset["train"].num_rows)
test_sample = dataset["train"][test_idx]["color"]

# Define transformation chain
transformation_chain = T.Compose(
    [
        T.Resize(int((256 / 224) * extractor.size["height"])),
        T.CenterCrop(extractor.size["height"]),
        T.ToTensor(),
        T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
    ]
)

# Extract embeddings for candidate_subset
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def extract_embedding(image):
    if image.mode != "RGB":
        image = image.convert("RGB")  # Convert grayscale to RGB
    image = transformation_chain(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image).last_hidden_state.mean(dim=1).squeeze().cpu()
    return embedding

candidate_embeddings = []
for image in tqdm(candidate_subset, desc="Extracting candidate embeddings"):
    embedding = extract_embedding(image)
    candidate_embeddings.append(embedding)

candidate_embeddings = torch.stack(candidate_embeddings)

# Extract embedding for test image
test_embedding = extract_embedding(test_sample)

# Compute cosine similarities
cosine_similarities = torch.nn.functional.cosine_similarity(
    test_embedding.unsqueeze(0), candidate_embeddings
)

# Find top 5 most similar images
top_k = 5
top_k_indices = torch.topk(cosine_similarities, top_k).indices

# Display the test image and top 5 similar images
fig, axes = plt.subplots(1, top_k + 1, figsize=(20, 5))
axes[0].imshow(np.array(test_sample))
axes[0].axis("off")
axes[0].set_title("Test Image")

for i, idx in enumerate(top_k_indices):
    axes[i + 1].imshow(np.array(candidate_subset[idx]))
    axes[i + 1].axis("off")
    axes[i + 1].set_title(f"Similar {i+1}")

plt.tight_layout()
plt.savefig("similar_images2.png")