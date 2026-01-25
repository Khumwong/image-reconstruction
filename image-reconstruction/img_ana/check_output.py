#!/usr/bin/env python3
"""
🔍 Check existing output folders and results

Run from anywhere - automatically searches parent folders
"""
import os
import sys
import glob
from pathlib import Path
import numpy as np


def format_size(bytes_val):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"


def check_folder(folder_path):
    """Check what's in an output folder"""
    folder = Path(folder_path)
    if not folder.exists():
        return None

    info = {
        'path': folder,
        'proton_paths': [],
        'images': [],
        'final_result': None,
        'total_size': 0
    }

    # Check proton_paths/
    pp_folder = folder / "proton_paths"
    if pp_folder.exists():
        info['proton_paths'] = list(pp_folder.glob("*.npy"))

    # Check images
    img_folder = folder / "proton_paths_images_2"
    if img_folder.exists():
        info['images'] = list(img_folder.glob("*.png"))

    # Check final result
    final_result = folder / "BackProjection_hull.npy"
    if final_result.exists():
        info['final_result'] = final_result

    # Calculate total size
    for root, dirs, files in os.walk(folder):
        for file in files:
            fp = os.path.join(root, file)
            if os.path.exists(fp):
                info['total_size'] += os.path.getsize(fp)

    return info


def main():
    print("="*70)
    print("🔍 Checking Output Folders")
    print("="*70)

    # Search in parent directory
    search_dir = Path(__file__).parent.parent  # hull/

    output_patterns = ["output*", "out*", "Output*"]

    found_folders = []
    for pattern in output_patterns:
        found_folders.extend(glob.glob(str(search_dir / pattern)))

    found_folders = sorted(set([f for f in found_folders if os.path.isdir(f)]))

    if not found_folders:
        print("\n❌ No output folders found")
        print(f"\n Searched in: {search_dir}")
        return

    print(f"\n✓ Found {len(found_folders)} output folder(s)\n")

    for i, folder_path in enumerate(found_folders, 1):
        print(f"{'─'*70}")
        print(f"📂 Folder {i}: {Path(folder_path).name}")
        print(f"{'─'*70}")

        info = check_folder(folder_path)
        if info is None:
            continue

        if info['proton_paths']:
            print(f"\n   📁 proton_paths/: {len(info['proton_paths'])} .npy files")

        if info['images']:
            print(f"   🖼️  images/: {len(info['images'])} .png files")

        if info['final_result']:
            print(f"\n   ⭐ BackProjection_hull.npy")
            try:
                data = np.load(info['final_result'])
                print(f"      Shape: {data.shape}")
                print(f"      Min: {data.min():.2f}, Max: {data.max():.2f}")
            except:
                pass

        print(f"\n   💾 Total: {format_size(info['total_size'])}\n")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
