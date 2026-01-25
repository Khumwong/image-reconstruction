#!/usr/bin/env python3
"""Test script to verify plotting fix"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.plotting import save_debug_overlay_back_rotated

# Load the back-projected data
output_path = Path("/home/sutpct/Workspace/santa/image-reconstruction/img_ana/output_reconstruction")

# Load BackProjection_hull.npy
print("Loading data...")
hull_data = np.load(output_path / "BackProjection_hull.npy")
print(f"Hull data shape: {hull_data.shape}")

# Create dummy inside/outside data (using zeros for quick test)
# In real scenario, these would be the actual reconstruction data
dummy_inside = np.zeros_like(hull_data, dtype=np.float32)
dummy_outside = np.zeros_like(hull_data, dtype=np.float32)

# Add some test pattern to see orientation
# Create horizontal stripes in Y direction
for y in range(100, 150, 10):
    dummy_inside[:, y, :] = 100

print("Generating test image...")
test_output = Path("/home/sutpct/Workspace/santa/image-reconstruction/img_ana/test_output")
test_output.mkdir(exist_ok=True)
(test_output / "Re_img_sum").mkdir(exist_ok=True)

save_debug_overlay_back_rotated(
    dummy_inside,
    dummy_outside,
    "test_orientation",
    test_output,
    num_pixels_z=512,
    l_mm=114.625,
    num_pixels_xy=512,
    hull_mask=hull_data,
    subfolder="Re_img_sum"
)

print(f"Test image saved to: {test_output}/Re_img_sum/test_orientation_debug_overlay.png")
print("\nCheck if horizontal stripes appear HORIZONTAL in the output image!")
