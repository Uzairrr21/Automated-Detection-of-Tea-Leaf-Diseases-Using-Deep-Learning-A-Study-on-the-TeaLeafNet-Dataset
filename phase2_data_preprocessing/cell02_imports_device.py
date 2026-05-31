# ==============================================================
# CELL 2 — Imports & Device Verification
# ==============================================================
import os, time, copy, random, math, warnings, struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import Counter
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import timm
 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
 
warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
 
# Enable cuDNN benchmark for faster GPU convolutions
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device       : {DEVICE}")
print(f"✅ PyTorch      : {torch.__version__}")
print(f"✅ TIMM         : {timm.__version__}")
 
if DEVICE.type == "cuda":
    print(f"✅ GPU          : {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA version : {torch.version.cuda}")
    try:
        t = torch.randn(2, 2, device=DEVICE)
        _ = t @ t
        print("✅ CUDA smoke test PASSED")
    except RuntimeError as e:
        print(f"❌ CUDA smoke test FAILED — falling back to CPU\n   {e}")
        DEVICE = torch.device("cpu")