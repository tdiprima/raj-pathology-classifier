# training script
import json
import os
import random
from glob import glob
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ImgDataset(Dataset):
    def __init__(self, root_dir, classes, img_size=(224, 224), augment=True):
        self.samples = []
        for cls in classes:
            files = glob(os.path.join(root_dir, cls, "*"))
            for f in files:
                self.samples.append((f, classes.index(cls)))
        self.img_size = img_size
        self.augment = augment
        self.transform_train = T.Compose(
            [
                T.Resize(img_size),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform_val = T.Compose(
            [
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform_train(img) if self.augment else self.transform_val(img)
        return img, label


def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0
    for imgs, lbls in tqdm(loader):
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, lbls)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


def validate(model, loader, loss_fn, device):
    model.eval()
    total = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, lbls)
            total += loss.item()
            correct += (preds.argmax(1) == lbls).sum().item()
            total_samples += imgs.size(0)
    return total / len(loader), correct / total_samples


def main():
    with open("../config.json", "r") as f:
        config = json.load(f)

    ROOT = config["root"]  # expects data/class/xxx.png
    CLASSES = sorted(
        [d.name for d in Path(ROOT).iterdir() if d.is_dir()]
    )  # auto-detect
    IMG_SIZE = tuple(config["img_size"])
    BATCH = config["batch_size"]
    EPOCHS = config["epochs"]
    LR = config["lr"]

    # build dataset
    dataset = ImgDataset(ROOT, CLASSES, img_size=IMG_SIZE, augment=True)
    random.shuffle(dataset.samples)
    split = int(config["train_split"] * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(0, split))
    val_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))
    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = models.resnet34(pretrained=True)
    model = models.resnet34(weights='IMAGENET1K_V1')
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
            torch.save(model.state_dict(), "best_resnet.pth")
            print("Saved best_resnet.pth")

    # export to ONNX
    dummy = torch.randn(1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device)
    model.load_state_dict(torch.load("best_resnet.pth"))
    model.eval()
    torch.onnx.export(model, dummy, "resnet.onnx", opset_version=11)
    print("Exported resnet.onnx")


if __name__ == "__main__":
    main()
