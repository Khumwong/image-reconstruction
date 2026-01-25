#!/usr/bin/env python3
"""Test plotting fix for multiple angles"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from visualization.plotting import save_debug_overlay_rotated_hull

output_orig = Path("/home/sutpct/Workspace/santa/image-reconstruction/img_ana/output_reconstruction")
output_fixed = Path("/home/sutpct/Workspace/santa/image-reconstruction/img_ana/output_reconstruction_fixed")

# Create output directories
(output_fixed / "average").mkdir(parents=True, exist_ok=True)

# Load hull
hull = np.load(output_orig / "BackProjection_hull.npy")

# Test angles
test_angles = [0, 102, 180]

for angle in test_angles:
    avg_file = output_orig / f"proton_paths/average_angle{angle}_degree.npy"
    if not avg_file.exists():
        print(f"Skipping angle {angle} (file not found)")
        continue

    print(f"\nProcessing angle {angle}...")
    avg_data = np.load(avg_file)

    save_debug_overlay_rotated_hull(
        avg_data,
        np.zeros_like(avg_data),
        f"average_angle{angle}_degree_FIXED",
        output_fixed,
        num_pixels_z=512,
        l_mm=114.625,
        num_pixels_xy=512,
        hull_mask=hull,
        subfolder="average"
    )
    print(f"  ✓ Saved: average_angle{angle}_degree_FIXED_debug_overlay.png")

print("\nDone! Check output_reconstruction_fixed/average/")
