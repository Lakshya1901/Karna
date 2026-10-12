"""Renders docs/bloom_comparison.png from the numbers printed by karna/benchmarks/benchmark.py."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 20,000 stored keys, 10,000 absent test keys, both filters targeting ~1% false positives
scenarios = ["Absent keys look different\n(order:, sess:, cart: ...)", "Absent keys look identical\n(user: ... same shape)"]
classic = {"fpr": [0.97, 0.88], "bytes": [23_963, 23_963]}
learned = {"fpr": [0.48, 1.01], "bytes": [2_452, 29_868]}

BLUE, GREY = "#2563eb", "#9ca3af"
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, key, title, unit in [
    (axes[0], "bytes", "Memory used (smaller is better)", "bytes"),
    (axes[1], "fpr", "False-positive rate (lower is better)", "%"),
]:
    x = range(len(scenarios))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], classic[key], w, label="Classic Bloom filter", color=GREY)
    b2 = ax.bar([i + w / 2 for i in x], learned[key], w, label="Learned Bloom filter (Karna)", color=BLUE)
    for bars in (b1, b2):
        for r in bars:
            v = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, v, f"{v:,.0f}" if unit == "bytes" else f"{v:.2f}%",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(classic[key] + learned[key]) * 1.18)
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Same job, same ~1% target: the learned filter wins big when it can learn, and says so when it can't",
             fontsize=11)
fig.tight_layout()
fig.savefig("docs/bloom_comparison.png", dpi=170)
print("saved docs/bloom_comparison.png")
