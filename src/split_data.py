# data split script
import random
import shutil
from pathlib import Path

ROOT = "data/classification"
OUT = "data_split"
VAL_RATIO = 0.2

classes = [d.name for d in Path(ROOT).iterdir() if d.is_dir()]
for cls in classes:
    imgs = list(Path(ROOT, cls).glob("*"))
    random.shuffle(imgs)
    split = int(len(imgs) * (1 - VAL_RATIO))
    train, val = imgs[:split], imgs[split:]
    for subset, files in [("train", train), ("val", val)]:
        outdir = Path(OUT, subset, cls)
        outdir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, outdir / f.name)

print("Done! train/val split in", OUT)
