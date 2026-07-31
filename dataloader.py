from huggingface_hub import constants
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import numpy as np
import matplotlib.pyplot as plt




print("Dataset dir: " + constants.HF_HUB_CACHE)
dataset = load_dataset('sayakpaul/nyu_depth_v2', trust_remote_code=True)


# rgb_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# gt_transform = transforms.Compose([
#     transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)      # Regular interpolation will blur the depth map and might create halo readings
# ])
#
# def apply_transform(batch):
#     transform_images = [rgb_transform(img.convert("RGB")) for img in batch["image"]]
#     transform_gt = []                   # Standard toTensor will normalize to 0 to 1, unapplicable
#     for gt in batch["depth_map"]:
#         gt_resized = gt_transform(gt)
#         gt_arr = np.array(gt_resized, dtype=np.float32)
#         gt_tensor = torch.from_numpy(gt_arr).unsqueeze(0)
#         transform_gt.append(gt_tensor)
#
#
#     return {
#         "image": transform_images,
#         "gt": transform_gt
#     }


def apply_transform(batch):
    transform_images = []
    transform_gt = []                   # Standard toTensor will normalize to 0 to 1, unapplicable
    for img, gt in zip(batch["image"], batch["depth_map"]):
        img = img.convert("RGB")

        # Eigen crop
        img = F.crop(img, top=44, left=40, height=427, width=561)
        gt = F.crop(gt, top=44, left=40, height=427, width=561)

        if torch.rand(1) > 0.5:
            img = F.hflip(img)
            gt = F.hflip(gt)

        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
        img = color_jitter(img)

        img = F.resize(img, (224, 224))
        gt = F.resize(gt, (224, 224), InterpolationMode.NEAREST)    # Regular interpolation will blur the depth map and might create halo readings

        img = transforms.ToTensor()(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        gt_arr = np.array(gt, dtype=np.float32)
        gt = torch.from_numpy(gt_arr).unsqueeze(0)

        transform_images.append(img)
        transform_gt.append(gt)

    return {
        "image": transform_images,
        "gt": transform_gt
    }


train_split = dataset["train"].with_transform(apply_transform)
validation_split = dataset["validation"].with_transform(apply_transform)


train_loader = DataLoader(
    train_split,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_split,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


def show_samples():
    batch = next(iter(train_loader))
    images = batch["image"]
    gts = batch["gt"]


    rnd = torch.randint(0, len(gts), (1,))
    img_tensor = images[rnd].squeeze().permute(1, 2, 0)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img_revert_norm = img_tensor * std + mean
    img_revert_norm = torch.clamp(img_revert_norm, 0, 1)

    plt.imshow(img_revert_norm.numpy())
    plt.show()
    plt.imshow(gts[rnd].squeeze().numpy(), cmap="plasma")
    plt.show()


# Testing
show_samples()
show_samples()
show_samples()
show_samples()
show_samples()

# for a in range(0, 5):
#     i = 0
#     for batch in train_loader:
#         images = batch["image"]
#         gt = batch["gt"]
#         print(a, "|", i, images.shape, gt.shape)
#         i += 1
#         if i > 20:
#             break

# No need to pre-shuffle before the main loop. Disk can handle by-epoch shuffles. Initial ~40 sec 100% load remains unchanged. Following epochs have 0.1 - 0.5 ms response time and ~23 - 30% load