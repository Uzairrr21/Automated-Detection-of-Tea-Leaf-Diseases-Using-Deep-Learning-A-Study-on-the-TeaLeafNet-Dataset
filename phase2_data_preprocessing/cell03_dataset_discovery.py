# ==============================================================
# CELL 3 — Dataset Discovery
# ==============================================================
DATA_ROOT = Path("/kaggle/input/datasets/harjindersinghdibru/tealeafnet/5000_tea_leaf_with_blackbg_geotagged")
 
CLASS_NAMES = sorted([
    d.name for d in DATA_ROOT.iterdir()
    if d.is_dir() and not d.name.startswith(".")
])
NUM_CLASSES  = len(CLASS_NAMES)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
 
print(f"📂 Classes : {CLASS_NAMES}")
print(f"📊 Mapping : {CLASS_TO_IDX}")