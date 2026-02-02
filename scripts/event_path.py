#!/usr/bin/env python3
"""
🎯 Process single event from angle 0 and save trajectory visualization

Usage:
    python event_path.py

This script will:
1. Find the first CSV file from angle 0
2. Process only that single event
3. Save visualization to ./output_reconstruction/event
"""
import os
import sys
import time
import glob
from pathlib import Path

# Add parent directory to path so we can import pct_reconstruction
sys.path.insert(0, str(Path(__file__).parent.parent))

from pct_reconstruction import HullImageReconstruction
from pct_reconstruction.core import get_mlp_parameters_rigorous, create_hull, rotate_hull
from pct_reconstruction.core.config import CSV_FOLDER, OUTPUT_FOLDER
from pct_reconstruction.processing import load_csv_fast, clean_data_batch, compute_mlp_rigorous, compute_average_image
from pct_reconstruction.visualization import (
    save_count_debug_overlay, save_wepl_debug_overlay, save_average_debug_overlay
)
from pct_reconstruction.utils import extract_angle_from_filename
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage import measure


def save_plotly_visualization(
    array_inside: np.ndarray,
    array_outside: np.ndarray,
    filename: str,
    output_path: Path,
    num_pixels_z: int,
    image_size_xy_mm: float,
    num_pixels_xy: int,
    hull_mask: np.ndarray = None,
    title: str = "Proton Paths"
):
    """Save interactive Plotly visualization of proton paths"""

    # Find best slice
    slice_sums_in = array_inside.sum(axis=(0, 1))
    slice_sums_out = array_outside.sum(axis=(0, 1))
    slice_sums_total = slice_sums_in + slice_sums_out
    v_position = int(slice_sums_total.argmax())

    # Get slices
    inside_slice = array_inside[:, :, v_position]
    outside_slice = array_outside[:, :, v_position]
    combined_slice = inside_slice + outside_slice

    # Transform (rotate 90 degrees)
    combined_slice_transformed = np.rot90(combined_slice, k=1)

    # Physical coordinates
    extent_x = [-57.3125, 57.3125]  # Y axis
    extent_y = [99-57.3125, 99+57.3125]  # X axis

    # Create coordinate arrays
    y_coords = np.linspace(extent_x[0], extent_x[1], combined_slice_transformed.shape[1])
    x_coords = np.linspace(extent_y[0], extent_y[1], combined_slice_transformed.shape[0])

    # Create figure
    fig = go.Figure()

    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=combined_slice_transformed,
        x=y_coords,
        y=x_coords,
        colorscale='Viridis',
        colorbar=dict(title=title),
        hovertemplate='Y: %{x:.2f} mm<br>X: %{y:.2f} mm<br>Value: %{z:.2f}<extra></extra>'
    ))

    # Add hull boundary if available
    if hull_mask is not None:
        hull_slice = hull_mask[:, :, v_position]

        if hull_slice.any():
            # Transform hull same way
            hull_slice_transformed = np.rot90(hull_slice, k=-1)
            hull_slice_transformed = np.fliplr(hull_slice_transformed)

            # Find contours
            contours = measure.find_contours(hull_slice_transformed, 0.5)

            pixel_size_xy = image_size_xy_mm / num_pixels_xy

            # Draw contours
            for contour in contours:
                row_indices = contour[:, 0]
                col_indices = contour[:, 1]

                # Map to physical coordinates
                y_mm = extent_x[0] + (col_indices * pixel_size_xy)
                x_mm = extent_y[1] - (row_indices * pixel_size_xy)

                fig.add_trace(go.Scatter(
                    x=y_mm,
                    y=x_mm,
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Phantom boundary',
                    showlegend=True,
                    hoverinfo='skip'
                ))

    # Add reference lines
    # Horizontal lines
    for y_val in [99, 49, 149]:
        fig.add_hline(y=y_val, line_dash="dash", line_color="red", opacity=0.5, line_width=1)

    # Vertical lines
    for x_val in [0, 50, -50]:
        fig.add_vline(x=x_val, line_dash="dash", line_color="red", opacity=0.5, line_width=1)

    # Update layout
    fig.update_layout(
        title=f'{title} - Slice {v_position}/{num_pixels_z}',
        xaxis_title='Y axis (mm)',
        yaxis_title='X axis (mm)',
        width=900,
        height=700,
        template='plotly_white',
        hovermode='closest'
    )

    # Save as HTML and PNG
    fig.write_html(output_path / f"{filename}.html")
    print(f"   Saved: {filename}.html")

    try:
        fig.write_image(output_path / f"{filename}.png", width=900, height=700)
        print(f"   Saved: {filename}.png")
    except Exception as e:
        print(f"   Warning: Could not save PNG (install kaleido: pip install kaleido): {e}")


def main():
    print("="*70)
    print("🎯 Single Event Path Visualization (Angle 0)")
    print("="*70)

    # ========== CONFIGURATION ==========
    # CSV_FOLDER and OUTPUT_FOLDER imported from config
    EVENT_OUTPUT_FOLDER = OUTPUT_FOLDER / "event"

    USE_HULL = True
    HULL_GEOMETRY = "oneFourth_cylinder"
    DEVICE = "cuda"
    RESOLUTION_XY = 115
    RESOLUTION_Z = 115

    # Physics parameters (same as run_reconstruction.py)
    Ein_MeV = 100.0
    d_mm = 25.0
    l_mm = 111.125 - 12.5 + 16
    l_cm = l_mm / 10
    image_size_xy_mm = 120.0  # ⭐ ROI size
    image_size_xy_cm = image_size_xy_mm / 10
    image_size_z_mm = 115.0
    # ===================================

    # 1. Check CSV folder
    print(f"\n📁 CSV Folder: {CSV_FOLDER}")
    if not os.path.exists(CSV_FOLDER):
        print(f"❌ Error: CSV folder not found!")
        return

    # 2. Find angle 0 files
    csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))

    angle_0_files = []
    for f in csv_files:
        angle = extract_angle_from_filename(Path(f))
        if angle == 0:
            angle_0_files.append(f)

    if len(angle_0_files) == 0:
        print(f"❌ Error: No angle 0 files found!")
        print(f"   Please check that CSV files exist with 'degree_0' in filename")
        return

    # Take first file only
    event_file = angle_0_files[0]
    print(f"\n✓ Found angle 0 file:")
    print(f"   {os.path.basename(event_file)}")

    # 3. Create output folders (organized by type)
    output_path = EVENT_OUTPUT_FOLDER
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path / "average", exist_ok=True)
    os.makedirs(output_path / "count", exist_ok=True)
    os.makedirs(output_path / "WEPL", exist_ok=True)
    print(f"\n📂 Output folder: {EVENT_OUTPUT_FOLDER}")

    # 4. Setup device
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        DEVICE = 'cpu'

    device = torch.device(DEVICE)
    print(f"\n⚙️  Device: {device}")

    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 5. Load MLP parameters
    print(f"\n🔧 Loading RIGOROUS MLP parameters...")
    t0 = time.time()
    mlp_parameters = get_mlp_parameters_rigorous(
        l_cm=l_cm,
        image_size_xy_cm=image_size_xy_cm,
        num_pixels=RESOLUTION_XY,
        recalculate=False,  # Use cache computed by reconstruction.py
        use_fast=False      # Use numerical integration (CORRECT)
    )
    print(f"   Done: {time.time()-t0:.1f}s")

    # 6. Create hull
    hull = None
    if USE_HULL:
        print(f"\n🏗️  Creating hull: {HULL_GEOMETRY}")
        l_cm = l_mm / 10
        h_cm = image_size_z_mm / 10

        hull = create_hull(
            geometry=HULL_GEOMETRY,
            num_pixels_xy=RESOLUTION_XY,
            num_pixels_z=RESOLUTION_Z,
            image_size_xy_mm=image_size_xy_mm,
            h_cm=h_cm,
            device=device
        )
        print(f"   Hull shape: {hull.shape}")

        # No rotation for angle 0
        hull_rot = hull
    else:
        hull_rot = None

    # 7. Transfer MLP parameters to GPU
    print(f"\n💾 Transferring MLP parameters to GPU...")
    mlp_params_gpu = {
        "Sigma1": torch.from_numpy(mlp_parameters["Sigma1"]).to(device).float(),
        "Sigma2": torch.from_numpy(mlp_parameters["Sigma2"]).to(device).float(),
        "R0": torch.from_numpy(mlp_parameters["R0"]).to(device).float(),
        "R1": torch.from_numpy(mlp_parameters["R1"]).to(device).float(),
        "X1": torch.from_numpy(mlp_parameters["X1"]).to(device).float()
    }
    print(f"   Done!")

    # 8. Process the event
    print(f"\n" + "="*70)
    print(f"🚀 PROCESSING EVENT")
    print("="*70)

    t_total = time.time()

    # Load CSV
    print(f"\n📖 Loading CSV...")
    t_load = time.time()
    df_full = load_csv_fast(Path(event_file))
    print(f"   Loaded: {len(df_full):,} rows in {time.time()-t_load:.2f}s")

    # Split by offset - select only middle offset
    unique_offsets = sorted(df_full['offset_mm'].unique())
    print(f"   Offsets found: {len(unique_offsets)} -> {unique_offsets}")

    # Select middle offset
    middle_idx = len(unique_offsets) // 2
    middle_offset = unique_offsets[middle_idx]
    print(f"   Selected middle offset: {middle_offset} mm (index {middle_idx}/{len(unique_offsets)-1})")

    # Keep only middle offset data
    df_middle = df_full[df_full['offset_mm'] == middle_offset]
    print(f"   Protons in middle offset: {len(df_middle):,}")

    # Randomly sample 30 protons
    NUM_SAMPLE = 10
    if len(df_middle) > NUM_SAMPLE:
        df_middle = df_middle.sample(n=NUM_SAMPLE, random_state=42)
        print(f"   Randomly sampled {NUM_SAMPLE} protons")
    else:
        print(f"   Using all {len(df_middle)} protons (less than {NUM_SAMPLE})")

    df_dict = {middle_offset: df_middle}

    # Initialize accumulators
    sum_wepl_gpu = torch.zeros(
        (RESOLUTION_XY, RESOLUTION_XY, RESOLUTION_Z),
        device=device, dtype=torch.float32
    )
    sum_count_gpu = torch.zeros_like(sum_wepl_gpu)

    debug_wepl_inside_gpu = torch.zeros_like(sum_wepl_gpu)
    debug_count_inside_gpu = torch.zeros_like(sum_wepl_gpu)
    debug_wepl_outside_gpu = torch.zeros_like(sum_wepl_gpu)
    debug_count_outside_gpu = torch.zeros_like(sum_wepl_gpu)

    # Batch process
    print(f"\n⚡ Processing protons...")
    t_proc = time.time()
    input_data_dict = clean_data_batch(df_dict, device)

    total_protons = 0
    for offset, input_data_gpu in input_data_dict.items():

        positions_v0 = input_data_gpu["positions_v0"]
        positions_v2 = input_data_gpu["positions_v2"]
        positions_v3 = input_data_gpu["positions_v3"]
        print("positions_v0:", positions_v0)

        # Normal reconstruction
        WEPL_img, count_img = compute_mlp_rigorous(
            input_data_gpu, mlp_params_gpu, hull_rot,
            l_mm, image_size_xy_mm, d_mm, image_size_z_mm,
            RESOLUTION_XY, RESOLUTION_Z, device
        )
        sum_wepl_gpu += WEPL_img
        sum_count_gpu += count_img

        # Debug visualization (separated inside/outside)
        if hull_rot is not None:
            WEPL_in, count_in, WEPL_out, count_out = compute_mlp_rigorous(
                input_data_gpu, mlp_params_gpu, hull_rot,
                l_mm, image_size_xy_mm, d_mm, image_size_z_mm,
                RESOLUTION_XY, RESOLUTION_Z, device,
                return_debug_images=True
            )
            debug_wepl_inside_gpu += WEPL_in
            debug_count_inside_gpu += count_in
            debug_wepl_outside_gpu += WEPL_out
            debug_count_outside_gpu += count_out

            del WEPL_in, count_in, WEPL_out, count_out

        total_protons += input_data_gpu["positions_u0"].shape[0]
        del WEPL_img, count_img

    print(f"   Processed {total_protons:,} protons in {time.time()-t_proc:.2f}s")

    # Compute averages
    print(f"\n📊 Computing averages...")
    avg_inside_gpu = compute_average_image(debug_wepl_inside_gpu, debug_count_inside_gpu)
    avg_outside_gpu = compute_average_image(debug_wepl_outside_gpu, debug_count_outside_gpu)
    avg_combined_gpu = compute_average_image(sum_wepl_gpu, sum_count_gpu)

    # Move to CPU for saving
    wepl_inside_np = debug_wepl_inside_gpu.cpu().numpy()
    count_inside_np = debug_count_inside_gpu.cpu().numpy()
    wepl_outside_np = debug_wepl_outside_gpu.cpu().numpy()
    count_outside_np = debug_count_outside_gpu.cpu().numpy()
    avg_inside_np = avg_inside_gpu.cpu().numpy()
    avg_outside_np = avg_outside_gpu.cpu().numpy()
    hull_rot_np = hull_rot.cpu().numpy() if hull_rot is not None else None

    # 9. Save results (organized by type)
    print(f"\n💾 Saving results...")

    # Save numpy arrays in respective folders
    np.save(output_path / "WEPL" / "WEPL_inside.npy", wepl_inside_np)
    np.save(output_path / "WEPL" / "WEPL_outside.npy", wepl_outside_np)
    np.save(output_path / "WEPL" / "WEPL_total.npy", wepl_inside_np + wepl_outside_np)

    np.save(output_path / "count" / "count_inside.npy", count_inside_np)
    np.save(output_path / "count" / "count_outside.npy", count_outside_np)
    np.save(output_path / "count" / "count_total.npy", count_inside_np + count_outside_np)

    np.save(output_path / "average" / "average_inside.npy", avg_inside_np)
    np.save(output_path / "average" / "average_outside.npy", avg_outside_np)

    if hull_rot_np is not None:
        np.save(output_path / "hull.npy", hull_rot_np)

    # Save visualizations
    print(f"\n🎨 Creating visualizations...")

    # Matplotlib PNG (debug overlays)
    print(f"\n   Creating matplotlib PNG overlays...")
    save_count_debug_overlay(
        count_inside_np,
        count_outside_np,
        "count_angle0",
        output_path,  # matplotlib function will add "count/" subfolder
        RESOLUTION_Z,
        image_size_xy_mm,
        RESOLUTION_XY,
        hull_rot_np
    )

    save_wepl_debug_overlay(
        wepl_inside_np,
        wepl_outside_np,
        "WEPL_angle0",
        output_path,  # matplotlib function will add "WEPL/" subfolder
        RESOLUTION_Z,
        image_size_xy_mm,
        RESOLUTION_XY,
        hull_rot_np
    )

    save_average_debug_overlay(
        avg_inside_np,
        avg_outside_np,
        "average_angle0",
        output_path,  # matplotlib function will add "average/" subfolder
        RESOLUTION_Z,
        image_size_xy_mm,
        RESOLUTION_XY,
        hull_rot_np
    )

    # Plotly interactive HTML
    print(f"\n   Creating plotly interactive HTML...")
    save_plotly_visualization(
        count_inside_np,
        count_outside_np,
        "count_angle0_interactive",
        output_path / "count",
        RESOLUTION_Z,
        image_size_xy_mm,
        RESOLUTION_XY,
        hull_rot_np,
        title="Proton Count"
    )

    save_plotly_visualization(
        wepl_inside_np,
        wepl_outside_np,
        "WEPL_angle0_interactive",
        output_path / "WEPL",
        RESOLUTION_Z,
        image_size_xy_mm,
        RESOLUTION_XY,
        hull_rot_np,
        title="WEPL (mm)"
    )

    save_plotly_visualization(
        avg_inside_np,
        avg_outside_np,
        "average_angle0_interactive",
        output_path / "average",
        RESOLUTION_Z,
        image_size_xy_mm,
        RESOLUTION_XY,
        hull_rot_np,
        title="Average WEPL (mm)"
    )

    elapsed = time.time() - t_total

    # 10. Summary
    print(f"\n{'='*70}")
    print(f"✅ EVENT PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"⏱️  Total time: {elapsed:.1f}s")
    print(f"📊 Protons processed: {total_protons:,}")
    print(f"\n📂 Output folder: {output_path.absolute()}")
    print(f"\n📁 Generated files by category:")

    # List files in each category
    for category in ["average", "count", "WEPL"]:
        cat_path = output_path / category
        if cat_path.exists():
            files = list(cat_path.glob("*"))
            if files:
                print(f"\n   {category}/:")
                for f in sorted(files):
                    size_mb = f.stat().st_size / 1e6
                    print(f"      {f.name} ({size_mb:.2f} MB)")

    # Hull file in root
    hull_file = output_path / "hull.npy"
    if hull_file.exists():
        size_mb = hull_file.stat().st_size / 1e6
        print(f"\n   hull.npy ({size_mb:.2f} MB)")

    print(f"\n💡 View results in Python:")
    print(f"   import numpy as np")
    print(f"   wepl = np.load('{output_path}/WEPL/WEPL_total.npy')")
    print(f"   print('Shape:', wepl.shape)")

    print(f"\n💡 View static PNG plots (matplotlib):")
    print(f"   - {output_path}/average/average_angle0_debug_overlay.png")
    print(f"   - {output_path}/count/count_angle0_debug_overlay.png")
    print(f"   - {output_path}/WEPL/WEPL_angle0_debug_overlay.png")

    print(f"\n💡 View interactive HTML plots (plotly):")
    print(f"   Open in browser for interactive zoom/pan:")
    print(f"   - {output_path}/average/average_angle0_interactive.html")
    print(f"   - {output_path}/count/count_angle0_interactive.html")
    print(f"   - {output_path}/WEPL/WEPL_angle0_interactive.html")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
