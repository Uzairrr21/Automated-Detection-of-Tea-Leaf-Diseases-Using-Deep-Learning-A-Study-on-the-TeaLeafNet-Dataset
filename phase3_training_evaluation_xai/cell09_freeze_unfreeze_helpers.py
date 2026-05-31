# ==============================================================
# CELL 9 — Freeze / Unfreeze Helpers
# ==============================================================
def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔒 Backbone FROZEN   — trainable params: {tr:,}")
 
def unfreeze_stage4(model):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.backbone.layers[3].parameters():
        p.requires_grad = True
    if hasattr(model.backbone, "norm"):
        for p in model.backbone.norm.parameters():
            p.requires_grad = True
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔓 Stage 4 UNFROZEN  — trainable params: {tr:,}")