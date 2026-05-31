# ==============================================================
# CELL 15 — Test Evaluation with TTA
# Change: N_TTA increased from 5 → 7 for tighter ensemble averaging
# Methodology unchanged: same Swin-T model, same weights, same logic
# ==============================================================
N_TTA = 7    # ← was 5; more views reduces variance at 95%+ accuracy

model.eval()
all_preds, all_targets = [], []

# Build TTA dataset
tta_ds = TTADataset(
    test_cache, all_labels[test_idx],
    tta_fn=tta_augment, n_tta=N_TTA
)

tta_loader = DataLoader(
    tta_ds, batch_size=16, shuffle=False,
    num_workers=0, pin_memory=(DEVICE.type == "cuda")
)

with torch.no_grad():
    for views, labels in tta_loader:
        # views shape: (B, N_TTA, C, H, W)
        B, N, C, H, W = views.shape
        views  = views.view(B * N, C, H, W).to(DEVICE, non_blocking=True)

        if DEVICE.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(views)          # (B*N, num_classes)
        else:
            logits = model(views)

        logits = logits.view(B, N, -1).mean(dim=1)   # average over TTA

        # ── FIXED: Bypassed PyTorch's C-level .numpy() wrapper using pure .tolist() ──
        preds  = logits.argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(labels.tolist())

all_preds   = np.array(all_preds)
all_targets = np.array(all_targets)

acc  = accuracy_score(all_targets, all_preds)
prec = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
rec  = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
f1   = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

print("\n" + "="*55)
print(f"  TEST SET PERFORMANCE  (TTA ×{N_TTA})")
print("="*55)
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print("="*55)
print(classification_report(all_targets, all_preds,
                             target_names=CLASS_NAMES, digits=4))