#!/usr/bin/env python3
"""
Example script demonstrating how to use the refactored img_ana module

This replaces the monolithic hull_fast_optimized.py with a clean modular approach.
"""
import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path so we can import img_ana
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the refactored module
from img_ana import HullImageReconstruction


def main():
    """Main function demonstrating module usage"""
    time_start = time.time()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="ULTRA-FAST hull-aware image reconstruction with rigorous physics (Modular Version)"
    )
    parser.add_argument(
        "--csv_folder",
        type=str,
        default="/home/sutpct/Workspace/test_pyeudaq_reader/ProcessedCSV",
        help="Folder containing CSV files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output_modular",
        help="Output folder"
    )
    parser.add_argument(
        "--no_hull",
        action="store_true",
        help="Disable hull geometry (default: use hull)"
    )
    parser.add_argument(
        "--hull_geometry",
        type=str,
        default="oneFourth_cylinder",
        choices=["oneFourth_cylinder", "full_cylinder"],
        help="Hull geometry type"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Computing device"
    )
    parser.add_argument(
        "--resolution_xy",
        type=int,
        default=512,
        help="XY resolution (default: 512)"
    )
    parser.add_argument(
        "--resolution_z",
        type=int,
        default=512,
        help="Z resolution (default: 512)"
    )
    parser.add_argument(
        "--init_angle",
        type=int,
        default=0,
        help="Initial angle offset"
    )

    args = parser.parse_args()

    # Prepare CSV paths
    processed_csv_folder = Path(args.csv_folder)
    csv_list = [f for f in os.listdir(processed_csv_folder) if f.endswith('.csv')]
    csv_paths = [os.path.join(processed_csv_folder, file) for file in csv_list]

    output_path = Path(args.output)
    os.makedirs(output_path, exist_ok=True)

    use_hull = not args.no_hull

    # Print configuration
    print(f"\n{'='*70}")
    print(f"ULTRA-FAST HULL-AWARE PROTON CT RECONSTRUCTION (MODULAR VERSION)")
    print(f"{'='*70}")
    print(f"Files: {len(csv_paths)}")
    print(f"Resolution: {args.resolution_xy}×{args.resolution_xy}×{args.resolution_z}")
    print(f"Hull support: {use_hull}")
    print(f"Hull geometry: {args.hull_geometry}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    # Create reconstruction object
    reconstructor = HullImageReconstruction(
        csv_paths=csv_paths,
        output_path=output_path,
        Ein_MeV=100.0,
        init_angle=args.init_angle,
        device=args.device,
        num_pixels_xy=args.resolution_xy,
        num_pixels_z=args.resolution_z
    )

    # Run reconstruction
    result = reconstructor.reconstruct(
        use_hull=use_hull,
        hull_geometry=args.hull_geometry
    )

    # Save final result
    np.save(output_path / "BackProjection_hull.npy", result)

    # Print completion message
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE!")
    print(f"   Time: {time.time()-time_start:.1f}s ({(time.time()-time_start)/60:.1f} min)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
