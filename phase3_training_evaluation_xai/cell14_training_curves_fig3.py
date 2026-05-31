# ==============================================================
# CELL 14 — Figure 3: Training Curves
# ==============================================================
df_h = pd.DataFrame(history)
df_h["global_epoch"] = range(1, len(df_h) + 1)
sep  = len(df_h[df_h["phase"].str.startswith("Phase 1")])
 
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Figure 3: Training Curves — Loss & Accuracy across Both Phases",
             fontsize=13, fontweight="bold")
 
for ax, (tr_col, vl_col), title, ylabel in zip(
    axes,
    [("train_loss","val_loss"), ("train_acc","val_acc")],
    ["Loss Curve", "Accuracy Curve"],
    ["Loss", "Accuracy"]
):
    ax.plot(df_h["global_epoch"], df_h[tr_col],
            label="Train", color="#2196F3", lw=2)
    ax.plot(df_h["global_epoch"], df_h[vl_col],
            label="Validation", color="#F44336", lw=2)
    ylo, yhi = ax.get_ylim()
    ax.axvline(sep + 0.5, color="#555", ls="--", lw=1.4, label="Phase boundary")
    ax.text(sep * 0.45, yhi * 0.97, "Phase 1",
            ha="center", fontsize=9, color="#555")
    ax.text(sep + (len(df_h)-sep)*0.5, yhi * 0.97, "Phase 2",
            ha="center", fontsize=9, color="#555")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Global Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
 
plt.tight_layout()
plt.savefig("fig3_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure 3 saved.")