"""Visualization functions for saving images and debug overlays"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from typing import Optional


def get_white_viridis_cmap():
    """Create a custom colormap: white → viridis (purple → yellow)

    Returns a colormap where the lowest values are white, then transitions
    smoothly into the standard viridis colormap.
    """
    viridis = plt.cm.viridis
    # Sample viridis colors
    viridis_colors = viridis(np.linspace(0, 1, 256))
    # Create new colormap starting with white
    white = np.array([1, 1, 1, 1])  # RGBA white
    # First ~5% is white, then blend into viridis
    n_white = 1  # ~1% of 256
    new_colors = np.vstack([
        np.tile(white, (n_white, 1)),
        viridis_colors[n_white:]
    ])
    return LinearSegmentedColormap.from_list('white_viridis', new_colors)


# ============================================================================
# Simple Image Functions (no debug overlay)
# ============================================================================

def save_reimg_image(
    array: np.ndarray,
    filename: str,
    output_path: Path,
    num_pixels_z: int,
    image_size_xy_mm: float,
    num_pixels_xy: int
) -> None:
    """Save reconstructed image (Re_img)

    Args:
        array: 3D reconstructed array to visualize
        filename: Output filename (without extension)
        output_path: Output directory path
        num_pixels_z: Number of pixels in Z
        image_size_xy_mm: Image ROI size in mm
        num_pixels_xy: Number of pixels in XY
    """
    slice_sums = array.sum(axis=(0, 1))
    v_position = int(slice_sums.argmax())

    slice_data = array[:, :, v_position]
    transformed_slice = np.rot90(slice_data, k=1)
    #transformed_slice = np.fliplr(transformed_slice)

    extent_horizontal = [-image_size_xy_mm/2, image_size_xy_mm/2]
    extent_vertical = [99-image_size_xy_mm/2, 99+image_size_xy_mm/2]

    plt.figure(figsize=(10, 8))
    im = plt.imshow(transformed_slice, cmap='gray',
                   extent=[extent_horizontal[0], extent_horizontal[1],
                          extent_vertical[0], extent_vertical[1]],
                   aspect='auto', origin='lower')

    plt.xlabel('Z axis (mm)')
    plt.ylabel('X axis (mm)')
    plt.title(f'Reconstructed Image - Slice {v_position}/{num_pixels_z}')
    plt.colorbar(im, label='Intensity')

    # plt.axhline(y=99, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    # plt.axhline(y=49, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    # plt.axhline(y=149, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    # plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    # plt.axvline(x=50, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    # plt.axvline(x=-50, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.savefig(output_path / "Re_img" / f"{filename}.png",
            bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close()


def save_reimg_sum_image(
    array: np.ndarray,
    filename: str,
    output_path: Path,
    num_pixels_z: int,
    image_size_xy_mm: float,
    num_pixels_xy: int
) -> None:
    """Save summed reconstructed image (Re_img_sum)

    Args:
        array: 3D summed reconstructed array to visualize
        filename: Output filename (without extension)
        output_path: Output directory path
        num_pixels_z: Number of pixels in Z
        image_size_xy_mm: Image ROI size in mm
        num_pixels_xy: Number of pixels in XY
    """
    slice_sums = array.sum(axis=(0, 1))
    v_position = int(slice_sums.argmax())

    slice_data = array[:, :, v_position]
    transformed_slice = np.rot90(slice_data, k=1)
    #transformed_slice = np.fliplr(transformed_slice)

    extent_horizontal = [-image_size_xy_mm/2, image_size_xy_mm/2]
    extent_vertical = [99-image_size_xy_mm/2, 99+image_size_xy_mm/2]

    plt.figure(figsize=(10, 7))
    im = plt.imshow(transformed_slice, cmap='gray',
                   extent=[extent_horizontal[0], extent_horizontal[1],
                          extent_vertical[0], extent_vertical[1]],
                   aspect='auto', origin='lower')

    plt.xlabel('Z axis (mm)')
    plt.ylabel('X axis (mm)')
    plt.title(f'Summed Reconstructed Image - Slice {v_position}/{num_pixels_z}')
    plt.colorbar(im, label='Intensity')

    plt.axhline(y=99, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axhline(y=49, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axhline(y=149, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(x=50, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(x=-50, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.savefig(output_path / "Re_img_sum" / f"{filename}.png",
            bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close()


# ============================================================================
# Debug Overlay Functions (with hull boundary)
# ============================================================================

def save_average_debug_overlay(
    array_inside: np.ndarray,
    array_outside: np.ndarray,
    filename: str,
    output_path: Path,
    num_pixels_z: int,
    image_size_xy_mm: float,
    num_pixels_xy: int,
    hull_mask: Optional[np.ndarray] = None
) -> None:
    """Save average WEPL debug overlay with hull boundary

    Args:
        array_inside: Inside hull average WEPL data
        array_outside: Outside hull average WEPL data
        filename: Output filename (without extension)
        output_path: Output directory path
        num_pixels_z: Number of pixels in Z
        image_size_xy_mm: Image ROI size in mm
        num_pixels_xy: Number of pixels in XY
        hull_mask: Rotated hull mask to draw boundary
    """
    slice_sums_in = array_inside.sum(axis=(0, 1))
    slice_sums_out = array_outside.sum(axis=(0, 1))
    slice_sums_total = slice_sums_in + slice_sums_out
    v_position = int(slice_sums_total.argmax())

    # Get both slices and combine
    inside_slice = array_inside[:, :, v_position]
    outside_slice = array_outside[:, :, v_position]
    combined_slice = inside_slice + outside_slice

    # Rotate to match expected orientation
    combined_slice_transformed = np.rot90(combined_slice, k=1)
    #combined_slice_transformed = np.fliplr(combined_slice_transformed)

    # Physical coordinates
    extent_horizontal = [-image_size_xy_mm/2, image_size_xy_mm/2]
    extent_vertical = [99-image_size_xy_mm/2, 99+image_size_xy_mm/2]

    plt.figure(figsize=(10, 8))

    # Show transformed proton paths with log scale
    # vmax = combined_slice_transformed.max()
    im = plt.imshow(combined_slice_transformed, cmap=get_white_viridis_cmap(),
                    extent=[extent_horizontal[0], extent_horizontal[1],
                           extent_vertical[0], extent_vertical[1]],
                    aspect='auto', origin='lower')

    plt.xlabel('Z axis (mm)')
    plt.ylabel('X axis (mm)')
    plt.title(f'Average WEPL\nSlice {v_position}/{num_pixels_z}')
    plt.colorbar(im, label='Average WEPL (mm)')
    plt.gca().set_xticks([-60, -40, -20, 0, 20, 40, 60])
    plt.gca().set_yticks([40, 60, 80, 100, 120, 140])
    plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(10))

    # Draw phantom mask boundary as contour
    if hull_mask is not None:
        from skimage import measure

        hull_slice = hull_mask[:, :, v_position]

        if hull_slice.any():
            # Apply SAME transformation to hull
            hull_slice_transformed = np.rot90(hull_slice, k=-1)  # 90 CW first
            hull_slice_transformed = np.fliplr(hull_slice_transformed)

            # Find contours
            contours = measure.find_contours(hull_slice_transformed, 0.5)

            pixel_size_xy = image_size_xy_mm / num_pixels_xy

            # Draw all contours
            for i, contour in enumerate(contours):
                row_indices = contour[:, 0]
                col_indices = contour[:, 1]

                # Map to physical coordinates
                y_mm = extent_horizontal[0] + (col_indices * pixel_size_xy)
                x_mm = extent_vertical[1] - (row_indices * pixel_size_xy)

                if i == 0:
                    plt.plot(y_mm, x_mm, 'k-', linewidth=0.5, label='Phantom boundary')
                else:
                    plt.plot(y_mm, x_mm, 'k-', linewidth=0.5)

            plt.legend(loc='upper right')

    # Crosshair at pivot point
    plt.axhline(y=99, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(y=49, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(y=149, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=50, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=-50, color='red', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.savefig(output_path / "average" / f"{filename}_debug_overlay.png",
            bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()


def save_count_debug_overlay(
    array_inside: np.ndarray,
    array_outside: np.ndarray,
    filename: str,
    output_path: Path,
    num_pixels_z: int,
    image_size_xy_mm: float,
    num_pixels_xy: int,
    hull_mask: Optional[np.ndarray] = None
) -> None:
    """Save proton count debug overlay with hull boundary

    Args:
        array_inside: Inside hull count data
        array_outside: Outside hull count data
        filename: Output filename (without extension)
        output_path: Output directory path
        num_pixels_z: Number of pixels in Z
        image_size_xy_mm: Image ROI size in mm
        num_pixels_xy: Number of pixels in XY
        hull_mask: Rotated hull mask to draw boundary
    """
    slice_sums_in = array_inside.sum(axis=(0, 1))
    slice_sums_out = array_outside.sum(axis=(0, 1))
    slice_sums_total = slice_sums_in + slice_sums_out
    v_position = int(slice_sums_total.argmax())

    # Get both slices and combine
    inside_slice = array_inside[:, :, v_position]
    outside_slice = array_outside[:, :, v_position]
    combined_slice = inside_slice + outside_slice

    # Rotate to match expected orientation
    combined_slice_transformed = np.rot90(combined_slice, k=1)
    #combined_slice_transformed = np.fliplr(combined_slice_transformed)

    # Physical coordinates
    extent_horizontal = [-image_size_xy_mm/2, image_size_xy_mm/2]
    extent_vertical = [99-image_size_xy_mm/2, 99+image_size_xy_mm/2]

    plt.figure(figsize=(10, 8))

    # Show transformed proton paths with log scale
    vmax = combined_slice_transformed.max()
    im = plt.imshow(combined_slice_transformed, cmap=get_white_viridis_cmap(),
                norm=SymLogNorm(linthresh=0.1, vmin=1, vmax=vmax if vmax > 0.1 else 10),
                extent=[extent_horizontal[0], extent_horizontal[1],
                       extent_vertical[0], extent_vertical[1]],
                aspect='auto', origin='lower')

    plt.xlabel('Z axis (mm)')
    plt.ylabel('X axis (mm)')
    plt.title(f'Proton Count\nSlice {v_position}/{num_pixels_z}')
    plt.colorbar(im, label='Count')
    plt.gca().set_xticks([-60, -40, -20, 0, 20, 40, 60])
    plt.gca().set_yticks([40, 60, 80, 100, 120, 140])
    plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(10))

    # Draw phantom mask boundary as contour
    if hull_mask is not None:
        from skimage import measure

        hull_slice = hull_mask[:, :, v_position]

        if hull_slice.any():
            # Apply SAME transformation to hull
            hull_slice_transformed = np.rot90(hull_slice, k=-1)
            hull_slice_transformed = np.fliplr(hull_slice_transformed)

            # Find contours
            contours = measure.find_contours(hull_slice_transformed, 0.5)

            pixel_size_xy = image_size_xy_mm / num_pixels_xy

            # Draw all contours
            for i, contour in enumerate(contours):
                row_indices = contour[:, 0]
                col_indices = contour[:, 1]

                # Map to physical coordinates
                y_mm = extent_horizontal[0] + (col_indices * pixel_size_xy)
                x_mm = extent_vertical[1] - (row_indices * pixel_size_xy)

                if i == 0:
                    plt.plot(y_mm, x_mm, 'k-', linewidth=0.5, label='Phantom boundary')
                else:
                    plt.plot(y_mm, x_mm, 'k-', linewidth=0.5)

            plt.legend(loc='upper right')

    # Crosshair at pivot point
    plt.axhline(y=99, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(y=49, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(y=149, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=50, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=-50, color='red', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.savefig(output_path / "count" / f"{filename}_debug_overlay.png",
            bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()


def save_wepl_debug_overlay(
    array_inside: np.ndarray,
    array_outside: np.ndarray,
    filename: str,
    output_path: Path,
    num_pixels_z: int,
    image_size_xy_mm: float,
    num_pixels_xy: int,
    hull_mask: Optional[np.ndarray] = None
) -> None:
    """Save WEPL debug overlay with hull boundary

    Args:
        array_inside: Inside hull WEPL data
        array_outside: Outside hull WEPL data
        filename: Output filename (without extension)
        output_path: Output directory path
        num_pixels_z: Number of pixels in Z
        image_size_xy_mm: Image ROI size in mm
        num_pixels_xy: Number of pixels in XY
        hull_mask: Rotated hull mask to draw boundary
    """
    slice_sums_in = array_inside.sum(axis=(0, 1))
    slice_sums_out = array_outside.sum(axis=(0, 1))
    slice_sums_total = slice_sums_in + slice_sums_out
    v_position = int(slice_sums_total.argmax())

    # Get both slices and combine
    inside_slice = array_inside[:, :, v_position]
    outside_slice = array_outside[:, :, v_position]
    combined_slice = inside_slice + outside_slice

    # Rotate to match expected orientation
    combined_slice_transformed = np.rot90(combined_slice, k=1)
    #combined_slice_transformed = np.fliplr(combined_slice_transformed)
    
    # Physical coordinates
    extent_horizontal = [-image_size_xy_mm/2, image_size_xy_mm/2]
    extent_vertical = [99-image_size_xy_mm/2, 99+image_size_xy_mm/2]

    plt.figure(figsize=(10, 8))

    # Show transformed proton paths with log scale
    vmax = combined_slice_transformed.max()
    im = plt.imshow(combined_slice_transformed, cmap=get_white_viridis_cmap(),
                norm=SymLogNorm(linthresh=0.1, vmin=1, vmax=vmax if vmax > 0.1 else 10),
                extent=[extent_horizontal[0], extent_horizontal[1],
                       extent_vertical[0], extent_vertical[1]],
                aspect='auto', origin='lower')

    plt.xlabel('Z axis (mm)')
    plt.ylabel('X axis (mm)')
    plt.title(f'WEPL\nSlice {v_position}/{num_pixels_z}')
    plt.colorbar(im, label='WEPL (mm)')
    plt.gca().set_xticks([-60, -40, -20, 0, 20, 40, 60])
    plt.gca().set_yticks([40, 60, 80, 100, 120, 140])
    plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(10))

    # Draw phantom mask boundary as contour
    if hull_mask is not None:
        from skimage import measure

        hull_slice = hull_mask[:, :, v_position]

        if hull_slice.any():
            # Apply SAME transformation to hull
            hull_slice_transformed = np.rot90(hull_slice, k=-1)
            hull_slice_transformed = np.fliplr(hull_slice_transformed)

            # Find contours
            contours = measure.find_contours(hull_slice_transformed, 0.5)

            pixel_size_xy = image_size_xy_mm / num_pixels_xy

            # Draw all contours
            for i, contour in enumerate(contours):
                row_indices = contour[:, 0]
                col_indices = contour[:, 1]

                # Map to physical coordinates
                y_mm = extent_horizontal[0] + (col_indices * pixel_size_xy)
                x_mm = extent_vertical[1] - (row_indices * pixel_size_xy)

                if i == 0:
                    plt.plot(y_mm, x_mm, 'k-', linewidth=0.5, label='Phantom boundary')
                else:
                    plt.plot(y_mm, x_mm, 'k-', linewidth=0.5)

            plt.legend(loc='upper right')

    # Crosshair at pivot point
    plt.axhline(y=99, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(y=49, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(y=149, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=50, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=-50, color='red', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.savefig(output_path / "WEPL" / f"{filename}_debug_overlay.png",
            bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()


def save_reimg_debug_overlay(
    array_inside: np.ndarray,
    array_outside: np.ndarray,
    filename: str,
    output_path: Path,
    num_pixels_z: int,
    image_size_xy_mm: float,
    num_pixels_xy: int,
    hull_mask: Optional[np.ndarray] = None,
    use_log_scale: bool = False
) -> None:
    """Save reconstructed image (Re_img) debug overlay with hull boundary

    Args:
        array_inside: Inside hull reconstructed data (back-rotated)
        array_outside: Outside hull reconstructed data (back-rotated)
        filename: Output filename (without extension)
        output_path: Output directory path
        num_pixels_z: Number of pixels in Z
        image_size_xy_mm: Image ROI size in mm
        num_pixels_xy: Number of pixels in XY
        hull_mask: Original hull mask (will be transformed to match data)
        use_log_scale: Whether to use log scale for color mapping
    """
    slice_sums_in = array_inside.sum(axis=(0, 1))
    slice_sums_out = array_outside.sum(axis=(0, 1))
    slice_sums_total = slice_sums_in + slice_sums_out
    v_position = int(slice_sums_total.argmax())

    # Get both slices and combine
    inside_slice = array_inside[:, :, v_position]
    outside_slice = array_outside[:, :, v_position]
    combined_slice = inside_slice + outside_slice

    # Rotate to match expected orientation
    combined_slice_transformed = np.rot90(combined_slice, k=1)
    #combined_slice_transformed = np.fliplr(combined_slice_transformed)

    # Physical coordinates
    extent_horizontal = [-image_size_xy_mm/2, image_size_xy_mm/2]
    extent_vertical = [99-image_size_xy_mm/2, 99+image_size_xy_mm/2]


    plt.figure(figsize=(10, 8))

    # Show transformed proton paths
    if use_log_scale:
        vmax = combined_slice_transformed.max()
        im = plt.imshow(combined_slice_transformed, cmap=get_white_viridis_cmap(),
                    norm=SymLogNorm(linthresh=0.1, vmin=1, vmax=vmax if vmax > 0.1 else 10),
                    extent=[extent_horizontal[0], extent_horizontal[1],
                           extent_vertical[0], extent_vertical[1]],
                    aspect='auto', origin='lower')
    else:
        im = plt.imshow(combined_slice_transformed, cmap=get_white_viridis_cmap(),
                    extent=[extent_horizontal[0], extent_horizontal[1],
                           extent_vertical[0], extent_vertical[1]],
                    aspect='auto', origin='lower')

    plt.xlabel('Z axis (mm)')
    plt.ylabel('X axis (mm)')
    plt.title(f'Reconstructed Image\nSlice {v_position}/{num_pixels_z}')
    plt.colorbar(im, label='Intensity (mm)')
    plt.gca().set_xticks([-60, -40, -20, 0, 20, 40, 60])
    plt.gca().set_yticks([40, 60, 80, 100, 120, 140])
    plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(10))

    # Draw phantom mask boundary as contour
    if hull_mask is not None:
        from skimage import measure

        hull_slice = hull_mask[:, :, v_position]

        if hull_slice.any():
            # Apply SAME transformation to hull
            hull_slice_transformed = np.rot90(hull_slice, k=2)
            hull_slice_transformed = np.rot90(hull_slice_transformed, k=2)
            hull_slice_transformed = np.fliplr(hull_slice_transformed)

            # Find contours
            contours = measure.find_contours(hull_slice_transformed, 0.5)

            pixel_size_xy = image_size_xy_mm / num_pixels_xy

            # Draw all contours
            for i, contour in enumerate(contours):
                row_indices = contour[:, 0]
                col_indices = contour[:, 1]

                # Map to physical coordinates
                y_mm = (col_indices) * pixel_size_xy + extent_horizontal[0]
                x_mm = (row_indices) * pixel_size_xy + extent_vertical[0]

                if i == 0:
                    plt.plot(y_mm, x_mm, 'k-', linewidth=0.5, label='Phantom boundary')
                else:
                    plt.plot(y_mm, x_mm, 'k-', linewidth=0.5)

            plt.legend(loc='upper right')

    # Crosshair at pivot point
    plt.axhline(y=99, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(y=49, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(y=149, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=50, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=-50, color='red', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.savefig(output_path / "Re_img" / f"{filename}_debug_overlay.png",
            bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()


def save_reimg_sum_debug_overlay(
    array_inside: np.ndarray,
    array_outside: np.ndarray,
    filename: str,
    output_path: Path,
    num_pixels_z: int,
    image_size_xy_mm: float,
    num_pixels_xy: int,
    hull_mask: Optional[np.ndarray] = None,
    use_log_scale: bool = False
) -> None:
    """Save summed reconstructed image (Re_img_sum) debug overlay with hull boundary

    Args:
        array_inside: Inside hull summed reconstructed data (back-rotated)
        array_outside: Outside hull summed reconstructed data (back-rotated)
        filename: Output filename (without extension)
        output_path: Output directory path
        num_pixels_z: Number of pixels in Z
        image_size_xy_mm: Image ROI size in mm
        num_pixels_xy: Number of pixels in XY
        hull_mask: Original hull mask (will be transformed to match data)
        use_log_scale: Whether to use log scale for color mapping
    """
    slice_sums_in = array_inside.sum(axis=(0, 1))
    slice_sums_out = array_outside.sum(axis=(0, 1))
    slice_sums_total = slice_sums_in + slice_sums_out
    v_position = int(slice_sums_total.argmax())

    # Get both slices and combine
    inside_slice = array_inside[:, :, v_position]
    outside_slice = array_outside[:, :, v_position]
    combined_slice = inside_slice + outside_slice

    # Rotate to match expected orientation
    combined_slice_transformed = np.rot90(combined_slice, k=1)
   #combined_slice_transformed = np.fliplr(combined_slice_transformed)

    # Physical coordinates
    extent_horizontal = [-image_size_xy_mm/2, image_size_xy_mm/2]
    extent_vertical = [99-image_size_xy_mm/2, 99+image_size_xy_mm/2]


    plt.figure(figsize=(10, 8))

    # Show transformed proton paths
    if use_log_scale:
        vmax = combined_slice_transformed.max()
        im = plt.imshow(combined_slice_transformed, cmap=get_white_viridis_cmap(),
                    norm=SymLogNorm(linthresh=0.1, vmin=1, vmax=vmax if vmax > 0.1 else 10),
                    extent=[extent_horizontal[0], extent_horizontal[1],
                           extent_vertical[0], extent_vertical[1]],
                    aspect='auto', origin='lower')
    else:
        im = plt.imshow(combined_slice_transformed, cmap=get_white_viridis_cmap(),
                    extent=[extent_horizontal[0], extent_horizontal[1],
                           extent_vertical[0], extent_vertical[1]],
                    aspect='auto', origin='lower')

    plt.xlabel('Z axis (mm)')
    plt.ylabel('X axis (mm)')
    plt.title(f'Summed Reconstructed Image\nSlice {v_position}/{num_pixels_z}')
    plt.colorbar(im, label='Intensity (mm)')
    plt.gca().set_xticks([-60, -40, -20, 0, 20, 40, 60])
    plt.gca().set_yticks([40, 60, 80, 100, 120, 140])
    plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(10))

    # Draw phantom mask boundary as contour
    if hull_mask is not None:
        from skimage import measure

        hull_slice = hull_mask[:, :, v_position]

        if hull_slice.any():
            # Apply SAME transformation to hull
            hull_slice_transformed = np.rot90(hull_slice, k=2)
            hull_slice_transformed = np.rot90(hull_slice_transformed, k=2)
            hull_slice_transformed = np.fliplr(hull_slice_transformed)

            # Find contours
            contours = measure.find_contours(hull_slice_transformed, 0.5)

            pixel_size_xy = image_size_xy_mm / num_pixels_xy

            # Draw all contours
            for i, contour in enumerate(contours):
                row_indices = contour[:, 0]
                col_indices = contour[:, 1]

                # Map to physical coordinates
                y_mm = (col_indices) * pixel_size_xy + extent_horizontal[0]
                x_mm = (row_indices) * pixel_size_xy + extent_vertical[0]

                if i == 0:
                    plt.plot(y_mm, x_mm, 'k-', linewidth=2, label='Phantom boundary')
                else:
                    plt.plot(y_mm, x_mm, 'k-', linewidth=2)

            plt.legend(loc='upper right')

    # Crosshair at pivot point
    plt.axhline(y=99, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(y=49, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axhline(y=149, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=50, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.axvline(x=-50, color='red', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.savefig(output_path / "Re_img_sum" / f"{filename}_debug_overlay.png",
            bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()


