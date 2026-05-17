import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


CLASS_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
}
CLASS_NAMES = ["sky", "tree", "road", "grass", "water", "building", "mount", "obj"]
NUM_CLASSES = 8


class StanfordBackgroundDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None, label_transform=None, split_ratio=0.8):
        self.data_dir = data_dir
        self.transform = transform
        self.label_transform = label_transform

        img_dir = os.path.join(data_dir, "iccv09Data", "images")
        lbl_dir = os.path.join(data_dir, "iccv09Data", "labels")

        all_images = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        np.random.seed(42)
        indices = np.random.permutation(len(all_images))

        train_count = int(len(all_images) * split_ratio)
        if split == "train":
            selected = indices[:train_count]
        elif split == "val":
            selected = indices[train_count:]
        else:
            selected = indices[train_count:]

        self.image_paths = [os.path.join(img_dir, all_images[i]) for i in selected]
        base_names = [os.path.splitext(all_images[i])[0] for i in selected]
        self.label_paths = [os.path.join(lbl_dir, f"{bn}.regions.txt") for bn in base_names]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label_matrix = np.loadtxt(self.label_paths[idx], dtype=np.int32)

        label_matrix = np.clip(label_matrix, 0, NUM_CLASSES - 1)

        label = Image.fromarray(label_matrix.astype(np.uint8), mode="L")

        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        else:
            label = torch.from_numpy(np.array(label)).long()

        return image, label


def get_transforms(img_size=320):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    label_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long()),
    ])

    return train_transform, val_transform, label_transform


def get_dataloaders(data_dir, batch_size=4, img_size=320):
    train_t, val_t, label_t = get_transforms(img_size)

    train_ds = StanfordBackgroundDataset(data_dir, "train", transform=train_t, label_transform=label_t)
    val_ds = StanfordBackgroundDataset(data_dir, "val", transform=val_t, label_transform=label_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader