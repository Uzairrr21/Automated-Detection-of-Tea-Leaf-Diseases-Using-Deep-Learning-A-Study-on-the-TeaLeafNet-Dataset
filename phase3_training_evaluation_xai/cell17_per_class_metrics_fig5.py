# ==============================================================
# CELL 17 — Figure 5: Per-Class Metrics
# ==============================================================
report = classification_report(all_targets, all_preds,
                                target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
mdf = pd.DataFrame({
    cls: {k: report[cls][k] for k in ["precision","recall","f1-score"]}
    for cls in CLASS_NAMES
}).T.rename(columns={"f1-score":"F1-Score",
                     "precision":"Precision",
                     "recall":"Recall"})
 
fig, ax = plt.subplots(figsize=(12, 5))
x, w = np.arange(len(CLASS_NAMES)), 0.25
for i, (col, color) in enumerate(zip(
    ["Precision","Recall","F1-Score"],
    ["#4CAF50","#2196F3","#FF9800"]
)):
    bars = ax.bar(x + i*w, mdf[col], width=w, label=col,
                  color=color, edgecolor="black", linewidth=0.6)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
 
ax.set_title("Figure 5: Per-Class Precision, Recall & F1-Score",
             fontsize=13, fontweight="bold")
ax.set_xticks(x + w)
ax.set_xticklabels(CLASS_NAMES, rotation=15)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fig5_per_class_metrics.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure 5 saved.")