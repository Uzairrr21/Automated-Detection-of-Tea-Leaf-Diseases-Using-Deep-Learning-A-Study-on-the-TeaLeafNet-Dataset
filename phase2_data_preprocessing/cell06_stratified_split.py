# ==============================================================
# CELL 6 — Stratified 70 / 15 / 15 Split
# ==============================================================
all_paths  = np.array(all_paths)
all_labels = np.array(all_labels)
 
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(sss1.split(all_paths, all_labels))
 
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_rel, test_rel = next(sss2.split(all_paths[temp_idx], all_labels[temp_idx]))
val_idx  = temp_idx[val_rel]
test_idx = temp_idx[test_rel]
 
print(f"📊 Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")
for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
    c = Counter(all_labels[idx])
    print(f"  {name}: " + " | ".join(
        f"{CLASS_NAMES[k]}={v}" for k, v in sorted(c.items())))