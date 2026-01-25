#!/usr/bin/env python3
"""Replot existing reconstruction data with fixed transformation"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from visualization.plotting import (
    save_debug_overlay_back_rotated,
    save_debug_overlay_rotated_hull,
    save_average_image
)

output_orig = Path("/home/sutpct/Workspace/santa/image-reconstruction/img_ana/output_reconstruction")
output_test = Path("/home/sutpct/Workspace/santa/image-reconstruction/img_ana/output_reconstruction_fixed")

# Create output directories
(output_test / "Re_img_sum").mkdir(parents=True, exist_ok=True)
(output_test / "average").mkdir(parents=True, exist_ok=True)

# Load hull
print("Loading hull...")
hull = np.load(output_orig / "BackProjection_hull.npy")
print(f"Hull shape: {hull.shape}")

# Load and replot Re_img_sum for angle 0
print("\nReplotting Re_img_sum angle 0...")
# We need to reconstruct the data from the arrays
# Let's use a simpler approach: just load and plot one of the saved npy files

# Load one of the proton path arrays
angle = 0
avg_file = output_orig / f"proton_paths/average_angle{angle}_degree.npy"
if avg_file.exists():
    print(f"Loading {avg_file}...")
    avg_data = np.load(avg_file)
    print(f"Average data shape: {avg_data.shape}")

    # For this test, treat all as "inside" and none as "outside"
    # In real scenario you'd need the actual separated data
    save_debug_overlay_rotated_hull(
        avg_data,  # inside
        np.zeros_like(avg_data),  # outside (dummy)
        f"average_angle{angle}_degree_FIXED",
        output_test,
        num_pixels_z=512,
        l_mm=114.625,
        num_pixels_xy=512,
        hull_mask=hull,  # This is the rotated hull for angle 0
        subfolder="average"
    )
    print(f"Saved to: {output_test}/average/average_angle{angle}_degree_FIXED_debug_overlay.png")

    # Also save grayscale
    save_average_image(
        avg_data,
        f"average_angle{angle}_degree_FIXED",
        output_test,
        num_pixels_z=512,
        l_mm=114.625,
        num_pixels_xy=512
    )
    print(f"Saved to: {output_test}/average/average_angle{angle}_degree_FIXED.png")
else:
    print(f"File not found: {avg_file}")

print("\nDone! Compare:")
print(f"  Original: {output_orig}/average/average_angle0_degree_debug_overlay.png")
print(f"  Fixed:    {output_test}/average/average_angle0_degree_FIXED_debug_overlay.png")
print(f"  Test ref: /home/sutpct/Workspace/santa/image-reconstruction/img_ana/output_reconstruction_test/average/average_angle0_degree_debug_overlay.png")
