# ==============================================================
# CELL 10 — MixUp Augmentation (training only)
#
# MixUp creates convex combinations of image pairs, forcing the
# model to learn smooth decision boundaries rather than memorizing
# per-image features. This is standard in competitive ML and
# widely cited (Zhang et al., 2018).
# ==============================================================
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
 
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)