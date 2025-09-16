import os
from glob import glob

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


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
