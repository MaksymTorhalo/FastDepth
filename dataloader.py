from huggingface_hub import constants
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import numpy as np




print("Dataset dir: " + constants.HF_HUB_CACHE)
dataset = load_dataset('sayakpaul/nyu_depth_v2', trust_remote_code=True)


rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gt_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)      # Regular interpolation will blur the depth map and might create halo readings
])

def apply_transform(batch):
    transform_images = [rgb_transform(img.convert("RGB")) for img in batch["image"]]
    transform_gt = []                   # Standard toTensor will normalize to 0 to 1, unapplicable
    for gt in batch["depth_map"]:
        gt_resized = gt_transform(gt)
        gt_arr = np.array(gt_resized, dtype=np.float32)
        gt_tensor = torch.from_numpy(gt_arr).unsqueeze(0)
        transform_gt.append(gt_tensor)


    return {
        "image": transform_images,
        "gt": transform_gt
    }


train_split = dataset["train"].shuffle(seed=42).with_transform(apply_transform)
validation_split = dataset["validation"].with_transform(apply_transform)


train_loader = DataLoader(
    train_split,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

validation_loader = DataLoader(
    validation_split,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


# Testing
# i = 0
# for batch in train_loader:
#     images = batch["image"]
#     gt = batch["gt"]
#     print(images.shape, gt.shape)
#     i += 1
#     if i > 100:
#         break