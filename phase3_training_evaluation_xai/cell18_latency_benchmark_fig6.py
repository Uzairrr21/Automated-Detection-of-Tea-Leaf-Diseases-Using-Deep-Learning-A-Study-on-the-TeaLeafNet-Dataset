# ==============================================================
# CELL 18 — Latency Benchmark & Figure 6
# ==============================================================
model.eval()
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
latencies   = []
 
for _ in range(10):
    with torch.no_grad():
        _ = model(dummy_input)
if DEVICE.type == "cuda":
    torch.cuda.synchronize()
 
for _ in range(100):
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_input)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    latencies.append((time.perf_counter() - t0) * 1000)
 
lat_mean = np.mean(latencies)
lat_std  = np.std(latencies)
lat_p95  = np.percentile(latencies, 95)
fps      = 1000.0 / lat_mean
 
print(f"  Mean latency : {lat_mean:.2f} ± {lat_std:.2f} ms")
print(f"  P95  latency : {lat_p95:.2f} ms")
print(f"  Throughput   : {fps:.1f} FPS  [{DEVICE}]")
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 6: Inference Latency — Single Image Benchmark",
             fontsize=13, fontweight="bold")
 
axes[0].hist(latencies, bins=25, color="#5C6BC0", edgecolor="black", lw=0.5)
axes[0].axvline(lat_mean, color="red",    ls="--", lw=2,
                label=f"Mean={lat_mean:.2f}ms")
axes[0].axvline(lat_p95,  color="orange", ls="--", lw=2,
                label=f"P95={lat_p95:.2f}ms")
axes[0].set_title("Latency Histogram")
axes[0].set_xlabel("ms"); axes[0].set_ylabel("Frequency")
axes[0].legend(); axes[0].grid(alpha=0.3)
 
axes[1].bar(["Mean±Std","P95"], [lat_mean, lat_p95],
            yerr=[lat_std, 0], color=["#5C6BC0","#EF5350"],
            capsize=8, edgecolor="black")
axes[1].set_ylabel("Latency (ms)")
axes[1].set_title(f"Summary — {fps:.1f} FPS on {DEVICE}")
axes[1].grid(axis="y", alpha=0.3)
 
plt.tight_layout()
plt.savefig("fig6_latency.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure 6 saved.")