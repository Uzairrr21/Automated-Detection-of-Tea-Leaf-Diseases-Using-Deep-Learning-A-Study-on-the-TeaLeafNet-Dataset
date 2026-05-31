# ==============================================================
# CELL 4 — File List & Figure 1: Class Distribution
# ==============================================================
all_paths, all_labels = [], []
for cls in CLASS_NAMES:
    for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG","*.JPEG"):
        for p in (DATA_ROOT / cls).glob(ext):
            all_paths.append(str(p))
            all_labels.append(CLASS_TO_IDX[cls])
 
print(f"📸 Total images : {len(all_paths)}")
 
counts = Counter(all_labels)
colors = ["#D64045", "#E98A15", "#5B8C5A", "#4D7EA8"][:NUM_CLASSES]
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 1: Dataset Class Distribution — TeaLeafNet",
             fontsize=14, fontweight="bold")
 
bars = axes[0].bar(CLASS_NAMES,
                   [counts[i] for i in range(NUM_CLASSES)],
                   color=colors, edgecolor="black", linewidth=0.8)
axes[0].set_title("Sample Count per Class", fontsize=12)
axes[0].set_xlabel("Disease Category")
axes[0].set_ylabel("Number of Images")
axes[0].tick_params(axis="x", rotation=20)
for bar, val in zip(bars, [counts[i] for i in range(NUM_CLASSES)]):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 3, str(val),
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
 
axes[1].pie([counts[i] for i in range(NUM_CLASSES)],
            labels=CLASS_NAMES, colors=colors, autopct="%1.1f%%",
            startangle=140, pctdistance=0.8,
            wedgeprops={"edgecolor":"white","linewidth":1.5})
axes[1].set_title("Proportional Distribution", fontsize=12)
 
plt.tight_layout()
plt.savefig("fig1_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure 1 saved.")