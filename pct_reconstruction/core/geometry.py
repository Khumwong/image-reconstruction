"""Hull geometry creation and transformations"""
import numpy as np
import torch
from typing import Literal


GeometryType = Literal['oneFourth_cylinder', 'full_cylinder']


def create_hull(
    geometry: GeometryType,
    num_pixels_xy: int,
    num_pixels_z: int,
    image_size_xy_mm: float,
    h_cm: float,
    device: torch.device,
    radius_cm: float = 5.0
) -> torch.Tensor:
    """Create 3D hull geometry for testing

    Args:
        geometry: Type of geometry ('oneFourth_cylinder' or 'full_cylinder')
        num_pixels_xy: Number of pixels in X and Y directions
        num_pixels_z: Number of pixels in Z direction
        l_cm: Length in cm (X, Y dimensions)
        h_cm: Height in cm (Z dimension)
        device: PyTorch device (CPU or CUDA)
        radius_cm: Cylinder radius in cm (default: 5.0)

    Returns:
        3D boolean tensor representing hull geometry

    Raises:
        ValueError: If geometry type is unknown
    """
    # Create coordinate grids
    image_size_xy_cm = image_size_xy_mm / 10.0
    y = np.linspace(-image_size_xy_cm/2, image_size_xy_cm/2, num_pixels_xy)
    x = np.linspace(-image_size_xy_cm/2, image_size_xy_cm/2, num_pixels_xy)
    z = np.linspace(-h_cm/2, h_cm/2, num_pixels_z)

    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')

    hull = np.full((num_pixels_xy, num_pixels_xy, num_pixels_z), False, dtype=bool)

    if geometry == 'oneFourth_cylinder':
        # Quarter cylinder: X <= 0, Y >= 0, X^2 + Y^2 <= R^2
        mask_quarter = (X <= 0) & (Y >= 0)
        mask_cylinder = (X**2 + Y**2) <= radius_cm**2
        hull[mask_quarter & mask_cylinder] = True

    elif geometry == 'full_cylinder':
        # Full cylinder: X^2 + Y^2 <= R^2
        mask_cylinder = (X**2 + Y**2) <= radius_cm**2
        hull[mask_cylinder] = True

    else:
        raise ValueError(f"Unknown geometry type: {geometry}")

    return torch.from_numpy(hull).to(device)


def rotate_hull(
    hull: torch.Tensor,
    angle_deg: float,
    device: torch.device
) -> torch.Tensor:
    """Rotate hull geometry around Z-axis

    Args:
        hull: 3D hull tensor (boolean)
        angle_deg: Rotation angle in degrees
        device: PyTorch device

    Returns:
        Rotated hull tensor (boolean)
    """
    from scipy.ndimage import rotate

    hull_np = hull.cpu().numpy()
    hull_rot_np = rotate(hull_np, angle=angle_deg, axes=(0, 1), reshape=False, order=0)
    hull_rot = torch.from_numpy(hull_rot_np).to(device).bool()

    return hull_rot
