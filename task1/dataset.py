import os
import scipy.io as sio
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FlowerDataset(Dataset):
    def __init__(self, root_dir, image_ids, labels, transform=None):
        self.root_dir = root_dir
        self.image_ids = image_ids
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_name = os.path.join(self.root_dir, f"image_{img_id:05d}.jpg")
        image = Image.open(img_name).convert("RGB")
        label = self.labels[idx] - 1
        if self.transform:
            image = self.transform(image)
        return image, label


def get_flower_datasets(data_dir):
    img_dir = os.path.join(data_dir, "102flowers", "jpg")
    labels_mat = sio.loadmat(os.path.join(data_dir, "102flowers", "imagelabels.mat"))
    setid_mat = sio.loadmat(os.path.join(data_dir, "102flowers", "setid.mat"))

    all_labels = labels_mat["labels"][0]
    train_ids = setid_mat["trnid"][0]
    val_ids = setid_mat["valid"][0]
    test_ids = setid_mat["tstid"][0]

    train_labels = [all_labels[i - 1] for i in train_ids]
    val_labels = [all_labels[i - 1] for i in val_ids]
    test_labels = [all_labels[i - 1] for i in test_ids]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = FlowerDataset(img_dir, train_ids, train_labels, transform=train_transform)
    val_ds = FlowerDataset(img_dir, val_ids, val_labels, transform=test_transform)
    test_ds = FlowerDataset(img_dir, test_ids, test_labels, transform=test_transform)

    return train_ds, val_ds, test_ds


def get_dataloaders(data_dir, batch_size=32):
    train_ds, val_ds, test_ds = get_flower_datasets(data_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader