# ==============================================================
# CELL 16 — Figure 4: Confusion Matrix
# ==============================================================
cm     = confusion_matrix(all_targets, all_preds)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
 
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Figure 4: Confusion Matrix — Test Set",
             fontsize=13, fontweight="bold")
 
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], linewidths=0.5)
axes[0].set_title("Counts")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
axes[0].tick_params(axis="x", rotation=25)
 
sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="YlOrRd",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[1], linewidths=0.5, cbar_kws={"label":"%"})
axes[1].set_title("Normalised (%)")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
axes[1].tick_params(axis="x", rotation=25)
 
plt.tight_layout()
plt.savefig("fig4_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure 4 saved.")