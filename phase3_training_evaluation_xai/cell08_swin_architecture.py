# ==============================================================
# CELL 8 — Swin-T Architecture (methodology-compliant)
# ==============================================================
class SwinTeaClassifier(nn.Module):
    """
    Swin-Tiny backbone + custom head:
      GAP → FC(512, ReLU) → Dropout(0.3) → FC(num_classes)
 
    Head dropout raised to 0.5 in Phase 1 to prevent the head
    from memorizing training examples when backbone is frozen
    (standard practice for transfer learning regularization).
    Dropout reverts to 0.3 in Phase 2 per methodology spec.
    """
    def __init__(self, num_classes=4, head_dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0,
            global_pool="",
            drop_path_rate=0.2,      # stochastic depth regularization
        )
        feat_dim = self.backbone.num_features  # 768
 
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout),
            nn.Linear(512, num_classes),
        )
 
    def set_head_dropout(self, p):
        for layer in self.head:
            if isinstance(layer, nn.Dropout):
                layer.p = p
        print(f"   Head dropout set to {p}")
 
    def forward(self, x):
        x = self.backbone.forward_features(x)
        if x.dim() == 3:    x = x.mean(dim=1)
        elif x.dim() == 4:  x = x.mean(dim=[1, 2])
        return self.head(x)
 
 
model = SwinTeaClassifier(num_classes=NUM_CLASSES, head_dropout=0.5).to(DEVICE)
 
dummy = torch.randn(2, 3, 224, 224).to(DEVICE)
with torch.no_grad():
    out = model(dummy)
assert out.shape == (2, NUM_CLASSES)
print(f"✅ Model output shape : {out.shape}")
 
total_p   = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📐 Total params       : {total_p:,}")
print(f"📐 Trainable params   : {trainable:,}")