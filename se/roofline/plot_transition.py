import csv
import os

import matplotlib.pyplot as plt

flops, ai, time_ms, gflops = [], [], [], []

csv_path = os.path.join(os.path.dirname(__file__), 'transition_results.csv')
if not os.path.exists(csv_path):
    csv_path = 'transition_results.csv'


with open(csv_path, newline='') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        if not any((v or '').strip() for v in row.values()):
            continue

        flops.append(float(row["FLOPs_per_element"]))
        ai.append(float(row["Arithmetic_Intensity"]))
        time_ms.append(float(row["Time_ms"]))
        gflops.append(float(row["GFLOPS"]))

baseline_time = time_ms[0]
time_ratio = [t / baseline_time for t in time_ms]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Time vs FLOPs
axes[0].plot(flops, time_ms, 'o-', linewidth=2, markersize=8, color='steelblue')
axes[0].set_xlabel('FLOPs per Element', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
axes[0].set_title('Time vs Computation Intensity', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=baseline_time, color='red', linestyle='--', alpha=0.5, label='Memory Copy Baseline')  # noqa
axes[0].legend()

# Plot 2: Time Ratio vs FLOPs
axes[1].plot(flops, time_ratio, 'o-', linewidth=2, markersize=8, color='darkgreen')
axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Memory Bound Threshold')
axes[1].axhline(y=1.1, color='orange', linestyle='--', alpha=0.5, label='Transition Zone')
axes[1].set_xlabel('FLOPs per Element', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Time Ratio (vs baseline)', fontsize=12, fontweight='bold')
axes[1].set_title('Memory Bound → Compute Bound Transition', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Plot 3: GFLOPS vs Arithmetic Intensity (linear scale)
axes[2].plot(ai, gflops, 'o-', linewidth=2, markersize=8, color='purple')
axes[2].set_xlabel('Arithmetic Intensity (FLOPS/Byte)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Performance (GFLOPS)', fontsize=12, fontweight='bold')
axes[2].set_title('Performance vs Arithmetic Intensity', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('memory_compute_transition.png', dpi=300, bbox_inches='tight')
print('Plot saved as memory_compute_transition.png')
plt.show()
