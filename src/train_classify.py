# training script
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

from utils.data_loader import create_data_loaders
from utils.dataset import ImgDataset
from utils.training import train_epoch, validate


def main():
    with open("../config.json", "r") as f:
        config = json.load(f)

    ROOT = os.path.join("..", config["root"])  # expects data/class/xxx.png
    CLASSES = sorted(
        [d.name for d in Path(ROOT).iterdir() if d.is_dir()]
    )  # auto-detect
    IMG_SIZE = tuple(config["img_size"])
    BATCH = config["batch_size"]
    EPOCHS = config["epochs"]
    LR = config["lr"]

    # build dataset
    dataset = ImgDataset(ROOT, CLASSES, img_size=IMG_SIZE, augment=True)
    train_loader, val_loader = create_data_loaders(
        dataset, config["train_split"], BATCH
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = models.resnet34(pretrained=True)
    model = models.resnet34(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, acc = validate(model, val_loader, loss_fn, device)
        print(
            f"Epoch {epoch}: train_loss {tr_loss:.4f} val_loss {val_loss:.4f} val_acc {acc:.4f}"
        )
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join("..", "best_resnet.pth"))
            print("Saved best_resnet.pth")

    # export to ONNX
    dummy = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device)
    model.load_state_dict(torch.load(os.path.join("..", "best_resnet.pth")))
    model.eval()
    torch.onnx.export(model, dummy, os.path.join("..", "resnet.onnx"), opset_version=11)
    print("Exported resnet.onnx")


if __name__ == "__main__":
    main()
