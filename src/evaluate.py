# evaluation script
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import PIL
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

class SafeImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            if os.path.isdir(cls_path):
                files = glob(os.path.join(cls_path, "*"))
                for f in files:
                    self.samples.append((f, self.classes.index(cls)))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        try:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
        except (OSError, IOError, PIL.UnidentifiedImageError) as e:
            print(f"Warning: Skipping corrupted image {p}: {e}")
            next_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(next_idx)

# ---- config ----
with open("../config.json", "r") as f:
    config = json.load(f)

VAL_DIR = os.path.join(
    "..", config["test_root"]
)  # or "data_split/val" if you used split_data.py
MODEL_PATH = os.path.join("..", "best_resnet.pth")
IMG_SIZE = tuple(config["img_size"])
BATCH = config["batch_size"]

# ---- dataset ----
transform = T.Compose(
    [
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_ds = SafeImageDataset(VAL_DIR, transform=transform)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- rebuild model ----
n_classes = len(val_ds.classes)
# model = models.resnet34(pretrained=False)
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, n_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ---- run evaluation ----
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ---- metrics ----
print("📊 Classification Report:")
print(
    classification_report(all_labels, all_preds, target_names=val_ds.classes, digits=3)
)

cm = confusion_matrix(all_labels, all_preds, labels=range(n_classes))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

# ---- confusion matrix plot ----
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    xticklabels=val_ds.classes,
    yticklabels=val_ds.classes,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
# plt.show()  # Commented out for no-GUI server
