#!/usr/bin/env python3
"""
🚀 Simple script to run hull reconstruction with pct_reconstruction

Usage:
    python run_reconstruction.py

This script will automatically:
1. Find all CSV files in the default folder
2. Create output folder if needed
3. Run reconstruction with optimal settings
4. Save results
"""
import os
import sys
import time
import glob
from pathlib import Path

# Add parent directory to path so we can import pct_reconstruction
sys.path.insert(0, str(Path(__file__).parent.parent))

from pct_reconstruction import HullImageReconstruction
from pct_reconstruction.core.config import (
    CSV_FOLDER, OUTPUT_FOLDER,
    USE_HULL, HULL_GEOMETRY,
    DEVICE, NUM_PIXELS_XY, NUM_PIXELS_Z,
    IMAGE_SIZE_XY_MM, IMAGE_SIZE_Z_MM,
    EIN_MEV, INIT_ANGLE, D_MM, L_MM,
    RADIUS_CM, BATCH_SIZE
)
import numpy as np


def main():
    print("="*70)
    print("🚀 Hull Image Reconstruction")
    print("="*70)

    # 1. Check CSV folder
    print(f"\n📁 CSV Folder: {CSV_FOLDER}")
    if not os.path.exists(CSV_FOLDER):
        print(f"❌ Error: CSV folder not found!")
        print(f"   Please check path: {CSV_FOLDER}")
        return

    # 2. Find CSV files
    csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))

    if len(csv_files) == 0:
        print(f"❌ Error: No CSV files found in {CSV_FOLDER}")
        return

    print(f"✓ Found {len(csv_files)} CSV files")

    # Show first few files
    print(f"\n📄 CSV Files (showing first 3):")
    for i, f in enumerate(csv_files[:3]):
        print(f"   {i+1}. {os.path.basename(f)}")
    if len(csv_files) > 3:
        print(f"   ... and {len(csv_files)-3} more files")

    # 3. Create output folder
    output_path = Path(OUTPUT_FOLDER)
    if output_path.exists():
        print(f"\n📂 Output folder exists: {OUTPUT_FOLDER}")
        response = input("   Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("❌ Cancelled by user")
            return
    else:
        os.makedirs(output_path, exist_ok=True)
        print(f"\n📂 Created output folder: {OUTPUT_FOLDER}")

    # 4. Show configuration
    print(f"\n⚙️  Configuration:")
    print(f"   Hull: {USE_HULL} ({HULL_GEOMETRY if USE_HULL else 'N/A'})")
    print(f"   Device: {DEVICE}")
    print(f"   Resolution: {NUM_PIXELS_XY}×{NUM_PIXELS_XY}×{NUM_PIXELS_Z}")
    print(f"   ROI Size: {IMAGE_SIZE_XY_MM}×{IMAGE_SIZE_XY_MM}×{IMAGE_SIZE_Z_MM} mm")
    print(f"   Energy: {EIN_MEV} MeV, Angle: {INIT_ANGLE}°")

    # 5. Create reconstructor
    print(f"\n🔧 Initializing reconstructor...")
    try:
        reconstructor = HullImageReconstruction(
            csv_paths=csv_files,
            output_path=output_path,
            Ein_MeV=EIN_MEV,
            init_angle=INIT_ANGLE,
            d_mm=D_MM,
            l_mm=L_MM,
            image_size_xy_mm=IMAGE_SIZE_XY_MM,
            image_size_z_mm=IMAGE_SIZE_Z_MM,
            num_pixels_xy=NUM_PIXELS_XY,
            num_pixels_z=NUM_PIXELS_Z,
            device=DEVICE,
            radius_cm=RADIUS_CM,
            batch_size=BATCH_SIZE
        )
        print("✓ Reconstructor initialized")
    except Exception as e:
        print(f"❌ Error initializing reconstructor: {e}")
        return

    # 6. Run reconstruction
    print(f"\n🚀 Starting reconstruction...")
    print(f"{'='*70}\n")

    time_start = time.time()

    try:
        result = reconstructor.reconstruct(
            use_hull=USE_HULL,
            hull_geometry=HULL_GEOMETRY
        )

        # Save final result
        result_file = output_path / "BackProjection_hull.npy"
        np.save(result_file, result)

        elapsed = time.time() - time_start

        # Success!
        print(f"\n{'='*70}")
        print(f"✅ RECONSTRUCTION COMPLETE!")
        print(f"{'='*70}")
        print(f"⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"\n📂 Output folder: {output_path.absolute()}")
        print(f"\n📁 Generated files:")

        # List output files
        proton_paths = output_path / "proton_paths"
        proton_images = output_path / "proton_paths_images_2"

        if proton_paths.exists():
            npy_files = list(proton_paths.glob("*.npy"))
            print(f"   proton_paths/: {len(npy_files)} .npy files")

        if proton_images.exists():
            png_files = list(proton_images.glob("*.png"))
            print(f"   proton_paths_images_2/: {len(png_files)} .png files")

        print(f"   BackProjection_hull.npy: Final result ({result.nbytes/1e6:.1f} MB)")

        print(f"\n💡 To view results:")
        print(f"   import numpy as np")
        print(f"   result = np.load('{result_file}')")
        print(f"   print(result.shape)")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Error during reconstruction:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
