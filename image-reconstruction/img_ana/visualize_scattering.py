#!/usr/bin/env python3
"""
📊 Visualize scattering matrices to verify they're computed correctly

Creates heatmap showing:
- Sigma values for all (i,j) pairs
- Should be zero for i >= j
- Should increase with distance
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pickle
import matplotlib.pyplot as plt

print("="*70)
print("📊 Visualizing Scattering Matrices")
print("="*70)

# Load rigorous MLP parameters
l_mm = 114.625
num_pixels = 512
filename = f"MLP_parameters_rigorous_l_{l_mm}_pixels_{num_pixels}.pkl"

if not Path(filename).exists():
    print(f"❌ File not found: {filename}")
    print("   Run reconstruction first to generate MLP parameters")
    sys.exit(1)

print(f"\n📂 Loading: {filename}")
with open(filename, "rb") as f:
    mlp_params = pickle.load(f)

Sigma1 = mlp_params["Sigma1"]
X1 = mlp_params["X1"]

print(f"✅ Loaded successfully")

# Compute Frobenius norm for each Sigma matrix
print(f"\n📐 Computing Frobenius norms...")
sigma_norms = np.zeros((num_pixels, num_pixels))

for i in range(num_pixels):
    for j in range(num_pixels):
        sigma_norms[i, j] = np.linalg.norm(Sigma1[i, j])

# ===================================================================
# PLOT 1: Full heatmap
# ===================================================================
print(f"📊 Creating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Full heatmap
ax = axes[0, 0]
im1 = ax.imshow(sigma_norms, origin='lower', cmap='viridis', aspect='auto')
ax.set_xlabel('End position j')
ax.set_ylabel('Start position i')
ax.set_title('Sigma Matrix Norms (Full View)\nShould be ZERO below diagonal')
plt.colorbar(im1, ax=ax, label='||Σ||')

# Add diagonal line
ax.plot([0, num_pixels], [0, num_pixels], 'r--', linewidth=2, label='Diagonal (i=j)')
ax.legend()

# Plot 2: Zoomed view (first 100x100)
ax = axes[0, 1]
zoom_size = 100
im2 = ax.imshow(sigma_norms[:zoom_size, :zoom_size], origin='lower', cmap='viridis', aspect='auto')
ax.set_xlabel('End position j')
ax.set_ylabel('Start position i')
ax.set_title(f'Sigma Matrix Norms (Zoom: first {zoom_size}×{zoom_size})')
plt.colorbar(im2, ax=ax, label='||Σ||')
ax.plot([0, zoom_size], [0, zoom_size], 'r--', linewidth=2)

# Plot 3: Cross-section (fix i, vary j)
ax = axes[1, 0]
i_fixed = 100
sigma_cross = sigma_norms[i_fixed, :]
ax.plot(sigma_cross, linewidth=2)
ax.axvline(i_fixed, color='r', linestyle='--', label=f'Start position i={i_fixed}')
ax.set_xlabel('End position j')
ax.set_ylabel('||Σ||')
ax.set_title(f'Scattering vs Distance (starting from i={i_fixed})\nShould be ZERO before i, INCREASE after i')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Scattering vs distance
ax = axes[1, 1]
distances = []
sigma_values = []

for j in range(i_fixed + 1, num_pixels, 10):  # Sample every 10 pixels
    dist_cm = X1[j] - X1[i_fixed]
    sigma_val = sigma_norms[i_fixed, j]
    distances.append(dist_cm)
    sigma_values.append(sigma_val)

ax.plot(distances, sigma_values, 'o-', linewidth=2, markersize=5)
ax.set_xlabel('Distance from start (cm)')
ax.set_ylabel('||Σ||')
ax.set_title(f'Scattering vs Physical Distance\nShould INCREASE with distance (Highland formula)')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_file = "scattering_matrices_visualization.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved plot to: {output_file}")

# ===================================================================
# Print statistics
# ===================================================================
print("\n" + "="*70)
print("📊 Statistics")
print("="*70)

# Count zeros in lower triangle
lower_triangle_mask = np.tril(np.ones((num_pixels, num_pixels)), k=0).astype(bool)
upper_triangle_mask = np.triu(np.ones((num_pixels, num_pixels)), k=1).astype(bool)

lower_vals = sigma_norms[lower_triangle_mask]
upper_vals = sigma_norms[upper_triangle_mask]

num_zeros_lower = np.sum(lower_vals < 1e-6)
num_nonzeros_upper = np.sum(upper_vals > 1e-6)

print(f"\nLower triangle (i ≥ j): should be ZERO")
print(f"  Zeros: {num_zeros_lower}/{len(lower_vals)} ({100*num_zeros_lower/len(lower_vals):.1f}%)")

print(f"\nUpper triangle (i < j): should be NON-ZERO")
print(f"  Non-zeros: {num_nonzeros_upper}/{len(upper_vals)} ({100*num_nonzeros_upper/len(upper_vals):.1f}%)")

print(f"\nScattering value range:")
print(f"  Min (upper triangle): {upper_vals.min():.6f}")
print(f"  Max (upper triangle): {upper_vals.max():.6f}")
print(f"  Mean (upper triangle): {upper_vals.mean():.6f}")

print("\n" + "="*70)
print("✅ Visualization complete!")
print(f"   Open: {output_file}")
print("="*70)

# Show plot
try:
    plt.show()
except:
    print("\n(Plot window not shown - display not available)")
