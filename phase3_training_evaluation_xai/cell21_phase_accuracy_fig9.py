# ==============================================================
# CELL 21 — Figure 9: Phase-wise Accuracy (CORRECTED)
# Dynamically reads N_TTA to prevent visual artifact errors
# ==============================================================
p1_best = max(df_h[df_h["phase"].str.startswith("Phase 1")]["val_acc"])
p2_best = max(df_h[df_h["phase"].str.startswith("Phase 2")]["val_acc"])

fig, ax = plt.subplots(figsize=(8, 5))

# Use dynamic N_TTA formatting instead of hardcoded strings
phases  = [
    "Phase 1\n(Feature\nExtraction)",
    "Phase 2\n(Fine-Tuning)",
    f"Test Acc\n(TTA ×{N_TTA})"
]
accs    = [p1_best, p2_best, acc]

bars = ax.bar(phases, accs,
              color=["#42A5F5","#66BB6A","#FFA726"],
              edgecolor="black", linewidth=0.8, width=0.45)

for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.004,
            f"{val:.4f}\n({val*100:.2f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylim(0, 1.15)
ax.set_ylabel("Accuracy")
ax.set_title("Figure 9: Best Val & Test Accuracy by Training Phase",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("fig9_phase_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure 9 saved successfully with dynamic evaluation tracking.")