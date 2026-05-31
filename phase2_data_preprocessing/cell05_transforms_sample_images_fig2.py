# ==============================================================
# CELL 5 — Transforms (numpy-free tensor conversion)
# ==============================================================
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

import struct

def pil_to_tensor_pure(pic):
    """PIL → float32 CHW tensor — completely numpy free."""
    if pic.mode not in ("RGB", "L"):
        pic = pic.convert("RGB")
    mode = pic.mode
    w, h  = pic.size
    raw   = pic.tobytes()
    n_ch  = 3 if mode == "RGB" else 1
    n_pix = h * w * n_ch
    flat  = struct.unpack(f"{n_pix}B", raw)
    t     = torch.tensor(flat, dtype=torch.float32)
    t     = t.view(h, w, n_ch).permute(2, 0, 1)
    t     = t.div(255.0)
    if n_ch == 1:
        t = t.expand(3, -1, -1)
    return t.contiguous()

# Train transform
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomGrayscale(p=0.10),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.Lambda(pil_to_tensor_pure),          # numpy-free ✅
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.10),
                             ratio=(0.3, 3.3), value=0),
])

# Val transform
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.Lambda(pil_to_tensor_pure),          # numpy-free ✅
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Test transform
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(pil_to_tensor_pure),          # numpy-free ✅
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# TTA transform
tta_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.Lambda(pil_to_tensor_pure),          # numpy-free ✅
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Figure 2: Sample images per class
fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(4 * NUM_CLASSES, 6))
fig.suptitle("Figure 2: Representative Tea Leaf Samples per Category",
             fontsize=13, fontweight="bold")

for col, cls in enumerate(CLASS_NAMES):
    cls_paths = [p for p, l in zip(all_paths, all_labels) if l == col]
    for row in range(min(2, len(cls_paths))):
        img = Image.open(cls_paths[row]).convert("RGB").resize((224, 224))
        ax  = axes[row][col] if NUM_CLASSES > 1 else axes[row]
        ax.imshow(img); ax.axis("off")
        if row == 0:
            ax.set_title(cls.replace("_", "\n"), fontsize=9,
                         fontweight="bold", color=colors[col])

plt.tight_layout()
plt.savefig("fig2_sample_images.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure 2 saved.")