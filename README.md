# raj-pathology-classifier

### 🧬 Overview

This repo contains a PyTorch implementation of a **10-class pathology image classifier**.  
The dataset consists of histopathology patches sorted into folders by tissue type (400px resolution).

Classes:

* Acinar tissue
* Dysplastic epithelium
* Fibrosis
* Lymph Aggregates
* Necrosis
* Nerves
* Normal ductal epithelium
* Reactive
* Stroma
* Tumor

The goal is to **recreate an earlier model in PyTorch**, train it on Raj's dataset, and evaluate performance with per-class metrics + confusion matrix.

---

### 📂 Project structure

```
raj-pathology-classifier/
├── data/                # raw data (NOT in git)
│   └── classification/  # 10 class folders
├── notebooks/           # experiments
│   └── explore_data.ipynb
├── src/                 # training & evaluation scripts
│   ├── train_classify.py
│   ├── evaluate.py
│   ├── split_data.py
│   ├── models/          # custom architectures
│   └── utils/           # helpers
├── outputs/             # trained weights, plots (gitignored)
├── config.json          # hyperparams
├── requirements.txt     # dependencies
└── README.md
```

---

### ⚙️ Setup

1. Clone the repo:

   ```bash
   git clone https://github.com/tdiprima/raj-pathology-classifier.git
   cd raj-pathology-classifier
   ```

2. Create a virtual env (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # (Linux/macOS)
   .venv\Scripts\activate      # (Windows)
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place dataset under:

   ```
   data/classification/400p-Acinar tissue/*.png
   data/classification/400p-Dysplastic epithelium/*.png
   ...
   ```

---

### 🚀 Training

Run:

```bash
python src/train_classify.py
```

* Automatically splits data into train/val (80/20).
* Saves best model to `models/DecaResNet.pth` (PyTorch model)
* Exports ONNX model to `models/DecaResNet.onnx` (ONNX model).
* TorchScript model => `models/DecaResnet.pt`

You can tweak hyperparams in `config.json`.

---

### 📊 Evaluation

After training, run:

```bash
python src/evaluate.py
```

* Prints per-class precision/recall/F1.
* Generates `confusion_matrix.png` in `outputs/`.

---

### 🧪 Experiments

Use `notebooks/explore_data.ipynb` for:

* Visual sanity checks of dataset
* Trying different augmentations
* Quick evaluation on small subsets

---

### 📌 Notes

* Dataset is too large for GitHub → keep it local or use DVC (Data Version Control) / symlinks if needed.
* All configs (paths, batch size, image size, epochs, LR) live in `config.json`.
* Outputs (weights, plots) are gitignored.

<br>
