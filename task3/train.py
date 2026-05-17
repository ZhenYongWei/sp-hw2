import os
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import get_dataloaders, NUM_CLASSES
from unet import UNet
from losses import DiceLoss, CombinedLoss
import swanlab


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def compute_miou(pred, target, num_classes=8):
    pred = pred.argmax(dim=1).cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    ious = []
    for cls in range(num_classes):
        intersection = ((pred == cls) & (target == cls)).sum()
        union = ((pred == cls) | (target == cls)).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def train_one_epoch(model, loader, criterion, optimizer, device, num_classes=8):
    model.train()
    total_loss = 0
    total_miou = 0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_miou += compute_miou(outputs, labels, num_classes) * images.size(0)
        count += images.size(0)
    return total_loss / count, total_miou / count


def evaluate(model, loader, criterion, device, num_classes=8):
    model.eval()
    total_loss = 0
    total_miou = 0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_miou += compute_miou(outputs, labels, num_classes) * images.size(0)
            count += images.size(0)
    return total_loss / count, total_miou / count


def run_unet_experiment(
    data_dir,
    loss_type="ce",
    batch_size=4,
    lr=1e-3,
    epochs=50,
    img_size=320,
    project="unet-segmentation",
    experiment_name=None,
    save_dir="checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)

    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size, img_size=img_size)

    if loss_type == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_type == "dice":
        criterion = DiceLoss(num_classes=NUM_CLASSES)
    elif loss_type == "combined":
        criterion = CombinedLoss(num_classes=NUM_CLASSES)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    if experiment_name is None:
        experiment_name = f"unet_loss={loss_type}_lr={lr}_epochs={epochs}"

    swanlab.init(
        project=project,
        experiment_name=experiment_name,
        mode="local",
        config={
            "model": "UNet",
            "loss_type": loss_type,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "img_size": img_size,
            "pretrained": False,
        },
    )

    best_val_miou = 0
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_miou = train_one_epoch(model, train_loader, criterion, optimizer, device, NUM_CLASSES)
        val_loss, val_miou = evaluate(model, val_loader, criterion, device, NUM_CLASSES)

        swanlab.log({
            "train_loss": train_loss,
            "train_miou": train_miou,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "epoch": epoch,
        })

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_miou={train_miou:.4f}, val_loss={val_loss:.4f}, val_miou={val_miou:.4f}")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), os.path.join(save_dir, f"{experiment_name}_best.pth"))

        scheduler.step()

    swanlab.finish()

    results = {
        "model": "UNet",
        "loss_type": loss_type,
        "lr": lr,
        "epochs": epochs,
        "best_val_miou": best_val_miou,
    }

    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{experiment_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def build_parser():
    parser = argparse.ArgumentParser(description="Task 3: U-Net segmentation training")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to the data directory")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["preset", "single"],
        default="preset",
        help="Run preset experiments or one custom experiment",
    )
    parser.add_argument("--loss-type", type=str, default="combined", choices=["ce", "dice", "combined"], help="Loss type")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--img-size", type=int, default=320, help="Input image size")
    parser.add_argument("--project", type=str, default="unet-segmentation", help="SwanLab project name")
    parser.add_argument("--experiment-name", type=str, default=None, help="Custom experiment name")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint output directory")
    return parser


def run_preset_experiments(data_dir):
    print("=" * 60)
    print("Experiment 1: UNet with Cross-Entropy Loss")
    print("=" * 60)
    run_unet_experiment(data_dir, loss_type="ce", epochs=20)

    print("=" * 60)
    print("Experiment 2: UNet with Dice Loss")
    print("=" * 60)
    run_unet_experiment(data_dir, loss_type="dice", epochs=20)

    print("=" * 60)
    print("Experiment 3: UNet with Combined Loss (CE + Dice)")
    print("=" * 60)
    run_unet_experiment(data_dir, loss_type="combined", epochs=20)

    print("All Task 3 experiments completed!")


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.mode == "preset":
        run_preset_experiments(args.data_dir)
    else:
        run_unet_experiment(
            args.data_dir,
            loss_type=args.loss_type,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            img_size=args.img_size,
            project=args.project,
            experiment_name=args.experiment_name,
            save_dir=args.save_dir,
        )
