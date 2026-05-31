# ==============================================================
# CELL 20 — Figure 8: Architecture Diagram
# ==============================================================
fig, ax = plt.subplots(figsize=(10, 13))
ax.axis("off"); fig.patch.set_facecolor("#FAFAFA")
 
blocks = [
    ("Input Image",              "224×224×3",                             "#90CAF9"),
    ("Patch Embedding",          "56×56 patches  |  dim=96",              "#CE93D8"),
    ("Swin Stage 1  🔒",         "56×56  |  dim=96   |  W-MSA",           "#A5D6A7"),
    ("Swin Stage 2  🔒",         "28×28  |  dim=192  |  SW-MSA",          "#A5D6A7"),
    ("Swin Stage 3  🔒",         "14×14  |  dim=384  |  W-MSA",           "#A5D6A7"),
    ("Swin Stage 4  ★ Unfrozen", "7×7    |  dim=768  |  SW-MSA",          "#FFA726"),
    ("Layer Norm  ← Grad-CAM",   "Final normalisation — XAI hook target",  "#FFA726"),
    ("Global Avg Pool (mean)",   "Token mean → 768-d vector",             "#EF9A9A"),
    ("FC  512  +  ReLU",         "512 units",                              "#80DEEA"),
    ("Dropout  0.3",             "Regularisation (0.5 in Phase 1)",        "#80DEEA"),
    ("FC  4  +  Softmax",        "BB  |  GL  |  RR  |  RSM",              "#FFCC80"),
]
 
ys = np.linspace(0.95, 0.04, len(blocks)); bh = 0.058
 
for i, (name, detail, color) in enumerate(blocks):
    y = ys[i]
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.08, y - bh/2), 0.84, bh,
        boxstyle="round,pad=0.01",
        facecolor=color, edgecolor="#333", linewidth=1.2,
        transform=ax.transAxes
    ))
    ax.text(0.5, y, f"{name}\n{detail}",
            ha="center", va="center", fontsize=8.5,
            transform=ax.transAxes,
            fontweight="bold" if ("★" in name or "Grad-CAM" in name)
                              else "normal")
    if i < len(blocks) - 1:
        ax.annotate("", xy=(0.5, ys[i+1] + bh/2 + 0.004),
                    xytext=(0.5, y - bh/2 - 0.004),
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
 
ax.set_title("Figure 8: Swin-T Architecture with Custom Classification Head",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(handles=[
    mpatches.Patch(color="#A5D6A7", label="Frozen (Phase 1 & 2)"),
    mpatches.Patch(color="#FFA726", label="Unfrozen Phase 2 — Stage 4 + Norm"),
    mpatches.Patch(color="#80DEEA", label="Custom head (both phases)"),
    mpatches.Patch(color="#EF9A9A", label="Global Avg Pooling"),
], loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=8.5)
 
plt.tight_layout()
plt.savefig("fig8_architecture.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure 8 saved.")