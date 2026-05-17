import os
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from dataset import get_dataloaders
from models import (
    get_resnet18, get_resnet34, get_resnet18_se,
    get_resnet18_cbam, get_vit_tiny, get_swin_tiny,
)
import swanlab


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def run_experiment(
    data_dir,
    model_name="resnet18",
    pretrained=True,
    num_classes=102,
    batch_size=32,
    lr=1e-3,
    fc_lr=1e-2,
    epochs=30,
    optimizer_name="sgd",
    scheduler_name="cosine",
    weight_decay=1e-4,
    project="flower-classification",
    experiment_name=None,
    save_dir="checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_factory = {
        "resnet18": get_resnet18,
        "resnet34": get_resnet34,
        "resnet18_se": get_resnet18_se,
        "resnet18_cbam": get_resnet18_cbam,
        "vit_tiny": get_vit_tiny,
        "swin_tiny": get_swin_tiny,
    }

    model = model_factory[model_name](num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()

    if pretrained and model_name in ["resnet18", "resnet34", "resnet18_se", "resnet18_cbam"]:
        fc_params = list(model.fc.parameters())
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
        optimizer = torch.optim.SGD(
            [{"params": backbone_params, "lr": lr}, {"params": fc_params, "lr": fc_lr}],
            momentum=0.9, weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    if experiment_name is None:
        experiment_name = f"{model_name}_pretrained={pretrained}_lr={lr}_epochs={epochs}"

    swanlab.init(
        project=project,
        experiment_name=experiment_name,
        mode="local",
        config={
            "model": model_name,
            "pretrained": pretrained,
            "batch_size": batch_size,
            "lr": lr,
            "fc_lr": fc_lr,
            "epochs": epochs,
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "weight_decay": weight_decay,
        },
    )

    best_val_acc = 0
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        swanlab.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch": epoch,
        })

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, f"{experiment_name}_best.pth"))

        if scheduler:
            scheduler.step()

    model.load_state_dict(torch.load(os.path.join(save_dir, f"{experiment_name}_best.pth")))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    swanlab.log({"test_loss": test_loss, "test_acc": test_acc})
    print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

    swanlab.finish()

    results = {
        "model": model_name,
        "pretrained": pretrained,
        "lr": lr,
        "fc_lr": fc_lr,
        "epochs": epochs,
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
    }

    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{experiment_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def build_parser():
    parser = argparse.ArgumentParser(description="Task 1: Flower classification training")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to the data directory")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["preset", "single"],
        default="preset",
        help="Run preset experiments or a single custom experiment",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet18_se", "resnet18_cbam", "vit_tiny", "swin_tiny"],
        help="Model name for single-experiment mode",
    )
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Disable pretrained weights")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for single-experiment mode")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for backbone or full model")
    parser.add_argument("--fc-lr", type=float, default=1e-2, help="Learning rate for classification head")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for single-experiment mode")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"], help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"], help="LR scheduler")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--project", type=str, default="flower-classification", help="SwanLab project name")
    parser.add_argument("--experiment-name", type=str, default=None, help="Custom experiment name")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint output directory")
    return parser


def run_preset_experiments(data_dir):
    print("=" * 60)
    print("Hyperparameter: lr=5e-4")
    print("=" * 60)
    run_experiment(data_dir, model_name="resnet18", pretrained=True, lr=5e-4, fc_lr=5e-3, epochs=30)

    print("=" * 60)
    print("Hyperparameter: lr=5e-3")
    print("=" * 60)
    run_experiment(data_dir, model_name="resnet18", pretrained=True, lr=5e-3, fc_lr=5e-2, epochs=30)

    print("=" * 60)
    print("Hyperparameter: epochs=50")
    print("=" * 60)
    run_experiment(data_dir, model_name="resnet18", pretrained=True, lr=1e-3, fc_lr=1e-2, epochs=50)

    print("=" * 60)
    print("Hyperparameter: batch_size=64")
    print("=" * 60)
    run_experiment(data_dir, model_name="resnet18", pretrained=True, lr=1e-3, fc_lr=1e-2, epochs=30, batch_size=64)

    print("=" * 60)
    print("Hyperparameter: ResNet-34 pretrained")
    print("=" * 60)
    run_experiment(data_dir, model_name="resnet34", pretrained=True, lr=1e-3, fc_lr=1e-2, epochs=30)

    print("All Task 1 experiments completed!")


if __name__ == "__main__":
    args = build_parser().parse_args()
    scheduler_name = None if args.scheduler == "none" else args.scheduler

    if args.mode == "preset":
        run_preset_experiments(args.data_dir)
    else:
        run_experiment(
            args.data_dir,
            model_name=args.model_name,
            pretrained=args.pretrained,
            batch_size=args.batch_size,
            lr=args.lr,
            fc_lr=args.fc_lr,
            epochs=args.epochs,
            optimizer_name=args.optimizer,
            scheduler_name=scheduler_name,
            weight_decay=args.weight_decay,
            project=args.project,
            experiment_name=args.experiment_name,
            save_dir=args.save_dir,
        )
