"""Draw assets/use_case_diagram.png (UML-style use case diagram) with matplotlib."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Circle

OUT = Path(__file__).resolve().parent.parent / "assets" / "use_case_diagram.png"
fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis("off")

ax.add_patch(Rectangle((3, 0.4), 7, 7.2, fill=False, lw=1.8, ec="#1f3b73"))
ax.text(6.5, 7.8, "System boundary: Student Service Request Monitoring System",
        ha="center", fontsize=12, weight="bold", color="#1f3b73")

use_cases = {
    "UC1": (6.5, 6.8, "Upload & Validate Dataset"),
    "UC2": (6.5, 5.8, "Explore Delay Analytics (EDA)"),
    "UC3": (6.5, 4.8, "Predict SLA Breach Risk"),
    "UC4": (6.5, 3.8, "View Prediction Explanation"),
    "UC5": (6.5, 2.8, "View SLA Dashboard & KPIs"),
    "UC6": (6.5, 1.8, "Review Flagged Requests"),
    "UC7": (6.5, 0.9, "Record Final Decision"),
}
for key, (x, y, label) in use_cases.items():
    red = key in ("UC6", "UC7")
    ax.add_patch(Ellipse((x, y), 4.4, 0.75, fc="#fdecec" if red else "#eaf1fb",
                         ec="#b3261e" if red else "#1f3b73", lw=1.4))
    ax.text(x, y, label, ha="center", va="center", fontsize=10)


def actor(x, y, name):
    ax.add_patch(Circle((x, y + 0.55), 0.22, fill=False, lw=1.6))
    ax.plot([x, x], [y + 0.33, y - 0.25], c="k", lw=1.6)
    ax.plot([x - 0.35, x + 0.35], [y + 0.15, y + 0.15], c="k", lw=1.6)
    ax.plot([x, x - 0.3], [y - 0.25, y - 0.8], c="k", lw=1.6)
    ax.plot([x, x + 0.3], [y - 0.25, y - 0.8], c="k", lw=1.6)
    ax.text(x, y - 1.1, name, ha="center", fontsize=11, weight="bold")


actor(1.3, 4.3, "Administrator")
actor(11.7, 3.3, "Service Officer")

for k in ["UC1", "UC2", "UC3", "UC5"]:  # administrator
    x, y, _ = use_cases[k]
    ax.plot([1.7, x - 2.2], [4.4, y], c="#444", lw=1)
for k in ["UC2", "UC3", "UC5", "UC6", "UC7"]:  # service officer
    x, y, _ = use_cases[k]
    ax.plot([11.3, x + 2.2], [3.4, y], c="#444", lw=1)

for a, b in [("UC3", "UC4"), ("UC6", "UC7")]:
    ax.annotate("", xy=(6.5, use_cases[b][1] + 0.38), xytext=(6.5, use_cases[a][1] - 0.38),
                arrowprops=dict(arrowstyle="->", ls="dashed", color="#b3261e"))
    ax.text(6.65, (use_cases[a][1] + use_cases[b][1]) / 2, "<<include>>", fontsize=8, color="#b3261e")

ax.text(6.5, 0.1, "Administrator: view results and audit log (read-only). Service Officer: review flags and record final decisions. "
        "Predictions never auto-deny a request.", ha="center", fontsize=8.5, style="italic")
fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
