# ==============================================================
# CELL 13 — Phase 2: Deep Fine-Tuning
# Key change: LR raised 2e-5 → 5e-5 so Stage 4 actually moves
# weights meaningfully — at 2e-5 with SGD the update was too
# small to shift val acc away from the memorized 99.7–100% plateau.
# MixUp alpha raised 0.08 → 0.15 for same anti-memorization reason.
# ==============================================================

unfreeze_stage4(model)

model.set_head_dropout(0.2)

model = run_phase(
    model,
    train_loader,
    val_loader,

    # Raised from 2e-5 — gives Stage 4 enough gradient signal
    # to reshape features toward the tea-leaf domain rather than
    # just oscillating at the Phase 1 plateau
    lr=5e-5,

    epochs=15,

    phase_name="Phase 2 — Deep Fine-Tuning",

    history=history,

    use_mixup=True,

    # Stronger MixUp than before — keeps train/val gap honest
    mixup_alpha=0.15,

    weight_decay=1e-5,

    warmup_epochs=2,

    optimizer_type="sgd"
)

torch.save(model.state_dict(), "swin_tea_best.pth")
print("✅ Checkpoint saved: swin_tea_best.pth")