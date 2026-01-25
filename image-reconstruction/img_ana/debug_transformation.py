#!/usr/bin/env python3
"""Debug transformation to understand the correct orientation"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
output_orig = Path("/home/sutpct/Workspace/santa/image-reconstruction/img_ana/output_reconstruction")
hull = np.load(output_orig / "BackProjection_hull.npy")
avg_data = np.load(output_orig / "proton_paths/average_angle0_degree.npy")

# Find best slice
v_position = int(avg_data.sum(axis=(0, 1)).argmax())
print(f"Best slice: {v_position}")

# Get slice
slice_data = avg_data[:, :, v_position]
hull_slice = hull[:, :, v_position]

print(f"Slice shape: {slice_data.shape}")
print(f"Hull slice shape: {hull_slice.shape}")

# Test different transformations
transformations = {
    "original": slice_data,
    "transpose": slice_data.T,
    "rot90_cw": np.rot90(slice_data, k=-1),
    "rot90_ccw": np.rot90(slice_data, k=1),
    "rot90_cw_fliplr": np.fliplr(np.rot90(slice_data, k=-1)),
    "flipud": np.flipud(slice_data),
    "fliplr": np.fliplr(slice_data),
}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, (name, transformed) in enumerate(transformations.items()):
    if idx >= len(axes):
        break

    ax = axes[idx]
    im = ax.imshow(transformed, cmap='gray', aspect='auto')
    ax.set_title(f'{name}\nShape: {transformed.shape}')
    ax.set_xlabel('Column (should be Y for horizontal paths)')
    ax.set_ylabel('Row (should be X)')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("transformation_debug.png", dpi=150)
print("Saved: transformation_debug.png")

# Analyze which direction has more variance (proton paths)
print("\nVariance analysis:")
print(f"Original - Row variance (X): {slice_data.var(axis=1).mean():.2f}")
print(f"Original - Col variance (Y): {slice_data.var(axis=0).mean():.2f}")
print(f"Transpose - Row variance (Y): {slice_data.T.var(axis=1).mean():.2f}")
print(f"Transpose - Col variance (X): {slice_data.T.var(axis=0).mean():.2f}")

print("\nProton paths should show HIGH variance along their direction!")
print("If Y-variance is higher -> paths run in Y -> need Y horizontal -> TRANSPOSE")
print("If X-variance is higher -> paths run in X -> need X horizontal -> NO TRANSPOSE")
