"""Back-projection and image rotation operations"""
import torch
import torch.nn.functional as F
import numpy as np


def rotate_image_gpu_batched(
    image_gpu: torch.Tensor,
    angle_deg: float,
    num_pixels_z: int,
    device: torch.device,
    batch_size: int = 64
) -> torch.Tensor:
    """GPU-accelerated image rotation using batched affine transformation

    Args:
        image_gpu: 3D image tensor [X, Y, Z]
        angle_deg: Rotation angle in degrees (negative for back-projection)
        num_pixels_z: Number of pixels in Z dimension
        device: PyTorch device
        batch_size: Batch size for processing slices

    Returns:
        Rotated 3D image tensor
    """
    # IMPORTANT: negate angle for back-projection (reverse rotation)
    angle_rad = angle_deg * np.pi / 180.0
    theta = torch.tensor([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                         [np.sin(angle_rad), np.cos(angle_rad), 0]],
                        dtype=torch.float32, device=device).unsqueeze(0)

    rotated_gpu = torch.zeros_like(image_gpu)

    for batch_idx in range(0, num_pixels_z, batch_size):
        end_z = min(batch_idx + batch_size, num_pixels_z)
        b_size = end_z - batch_idx

        # Extract slices and prepare for affine transformation
        slices = image_gpu[:, :, batch_idx:end_z].permute(2, 0, 1).unsqueeze(1)

        # Create affine grid
        grid = F.affine_grid(theta.expand(b_size, -1, -1), slices.size(), align_corners=False)

        # Apply rotation
        rotated = F.grid_sample(slices, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Store result
        rotated_gpu[:, :, batch_idx:end_z] = rotated.squeeze(1).permute(1, 2, 0)

    return rotated_gpu


def accumulate_projection(
    sum_wepl_gpu: torch.Tensor,
    sum_count_gpu: torch.Tensor,
    voxel_indices: dict,
    num_pixels_xy: int,
    num_pixels_z: int
) -> None:
    """Accumulate WEPL values into 3D image using scatter operations

    Args:
        sum_wepl_gpu: Accumulated WEPL tensor [X, Y, Z] (modified in-place)
        sum_count_gpu: Accumulated count tensor [X, Y, Z] (modified in-place)
        voxel_indices: Dict with keys x_idx, y_idx, z_idx, wepl
        num_pixels_xy: Number of pixels in XY
        num_pixels_z: Number of pixels in Z
    """
    if voxel_indices is None:
        return

    # Convert 3D indices to linear indices
    linear_indices = (voxel_indices['x_idx'] * num_pixels_xy * num_pixels_z +
                     voxel_indices['y_idx'] * num_pixels_z +
                     voxel_indices['z_idx'])

    # Scatter add WEPL values
    sum_wepl_gpu.view(-1).scatter_add_(0, linear_indices, voxel_indices['wepl'])

    # Scatter add counts
    sum_count_gpu.view(-1).scatter_add_(0, linear_indices,
                                        torch.ones_like(voxel_indices['wepl']))


def compute_average_image(
    sum_wepl_gpu: torch.Tensor,
    sum_count_gpu: torch.Tensor
) -> torch.Tensor:
    """Compute average WEPL image from accumulated values

    Args:
        sum_wepl_gpu: Accumulated WEPL tensor
        sum_count_gpu: Accumulated count tensor

    Returns:
        Average WEPL image (same shape as input)
    """
    average_gpu = torch.where(sum_count_gpu != 0,
                             sum_wepl_gpu / sum_count_gpu,
                             torch.zeros_like(sum_wepl_gpu))
    return average_gpu
