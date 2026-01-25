#!/usr/bin/env python3
"""Test flipud transformation"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from visualization.plotting import save_debug_overlay_rotated_hull

output_test = Path("/home/sutpct/Workspace/santa/image-reconstruction/img_ana/output_reconstruction_test")
output_fixed = Path("/home/sutpct/Workspace/santa/image-reconstruction/img_ana/output_reconstruction_fixed")

# Create output directories
(output_fixed / "average").mkdir(parents=True, exist_ok=True)

# Load hull from test directory
hull = np.load(output_test / "BackProjection_hull.npy")

# Load angle 0 data
angle = 0
avg_data = np.load(output_test / f"proton_paths/average_angle{angle}_degree.npy")

print(f"Processing angle {angle} with FLIPUD...")
save_debug_overlay_rotated_hull(
    avg_data,
    np.zeros_like(avg_data),
    f"average_angle{angle}_degree_FLIPUD",
    output_fixed,
    num_pixels_z=512,
    l_mm=114.625,
    num_pixels_xy=512,
    hull_mask=hull,
    subfolder="average"
)
print(f"✓ Saved: average_angle{angle}_degree_FLIPUD_debug_overlay.png")

# Also test angle 102
angle = 102
avg_data = np.load(output_test / f"proton_paths/average_angle{angle}_degree.npy")

print(f"Processing angle {angle} with FLIPUD...")
save_debug_overlay_rotated_hull(
    avg_data,
    np.zeros_like(avg_data),
    f"average_angle{angle}_degree_FLIPUD",
    output_fixed,
    num_pixels_z=512,
    l_mm=114.625,
    num_pixels_xy=512,
    hull_mask=hull,
    subfolder="average"
)
print(f"✓ Saved: average_angle{angle}_degree_FLIPUD_debug_overlay.png")

print("\nCompare:")
print("  Test (correct):  output_reconstruction_test/average/")
print("  Fixed (flipud):  output_reconstruction_fixed/average/")
