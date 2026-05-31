# ==============================================================
# CELL 12 — Phase 1: Feature Extraction
# Key change: LR reduced 3e-4 → 1e-4 to slow head convergence
# so val acc cannot race ahead of train acc and hit 100% by ep 4.
# More MixUp alpha (0.1→0.2) adds inter-sample mixing that makes
# validation harder to memorize even with random crops.
# ==============================================================

history = {k: [] for k in
           ["phase","epoch","train_loss","val_loss","train_acc","val_acc"]}

freeze_backbone(model)

model.set_head_dropout(0.3)

model = run_phase(
    model,
    train_loader,
    val_loader,

    # Reduced from 3e-4 — prevents head from converging so fast
    # that val memorization occurs before train accuracy catches up
    lr=1e-4,

    epochs=8,

    phase_name="Phase 1 — Feature Extraction",

    history=history,

    use_mixup=True,

    # Stronger MixUp — blended samples are harder to memorize on val
    mixup_alpha=0.2,

    weight_decay=1e-4,

    warmup_epochs=2,

    optimizer_type="adamw"
)