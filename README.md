# How to use MMUZC

**MMUzC** (Multi-Modal Urban Zone Classifier) is a lightweight, few-shot learning framework designed for *fine-grained Urban Functional Zone (UFZ)* classification by fusing **remote-sensing imagery** and **street-view panoramas**.
It integrates a **Swin Transformer** backbone for aerial cues, a **PIDNet** branch for ground-level semantics, and a **LoRA-Fusion** block that adaptively weights cross-modal embeddings.

## 🛠 Environment Setup

Verified configuration (CUDA 12.9):

```bash
# create & activate env
conda create -n mmuzc python=3.12 -y
conda activate mmuzc

# CUDA 12.9 + PyTorch 2.7
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu121
pip install torchgeo==0.71

# core libs
pip install timm==0.9.16
pip install matplotlib opencv-python numpy pandas

# utilities
pip install openpyxl tqdm tensorboard
```

## 📦 Data Download

The training/validation data (remote-sensing embeddings, labels, and segmentation maps) are packed in `data.zip`.
You can download it from Baidu NetDisk:

**Link:** https://pan.baidu.com/s/1WoUEEi1-5jpafnouybKUIw
**Extraction code:** `rg3z`

After downloading, unzip the archive and place the `data/` folder in the repository root (no extra path changes are needed).

Put the remote-sensing image embedding result (`image_embeddings04271_swinT.csv`), the urban-functional-zone label (`label.xlsx`), and the segmentation results (`output.csv`) in the `data/` folder.

To train the model, run:

```
python train_model.py
```

The trained weights will be saved to model.pth.

To draw loss, run

```
python drawloss.py
```

The proposed LoRA-Fusion module is implemented in `exmodel.py`.
