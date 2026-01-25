"""Trajectory calculations including straight paths and MLP computations"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from ..utils.helpers import index_to_position


def compute_straight_trajectory_batch(
    p_b: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    angle: Tuple[torch.Tensor, torch.Tensor],
    N: int,
    l_cm: float,
    h_cm: float,
    num_pixels_xy: int,
    num_pixels_z: int,
    device: torch.device
) -> torch.Tensor:
    """GPU-accelerated straight trajectory calculation (batched)

    Args:
        p_b: Tuple of (px, py, pz) position tensors
        angle: Tuple of (angle_y, angle_z) angle tensors
        N: Number of protons
        l_cm: Length in cm (x, y)
        h_cm: Height in cm (z)
        num_pixels_xy: Number of pixels in x, y
        num_pixels_z: Number of pixels in z
        device: PyTorch device

    Returns:
        Trajectory indices tensor [proton_id, x, y, z]
    """
    pixel_size_xy = l_cm / num_pixels_xy
    pixel_size_z = h_cm / num_pixels_z

    px, py, pz = p_b

    # Create x positions
    x = torch.linspace(-l_cm/2 + pixel_size_xy/2,
                      l_cm/2 - pixel_size_xy/2,
                      num_pixels_xy, device=device)

    x_exp = x.unsqueeze(1).expand(-1, N)

    # Calculate positions along beam path
    x_position = x_exp
    y_position = torch.tan(angle[0]) * (x_exp - px) + py
    z_position = torch.tan(angle[1]) * (x_exp - px) + pz

    # Convert to indices
    x_index = ((x_position + l_cm/2) / pixel_size_xy).long()
    y_index = ((y_position + l_cm/2) / pixel_size_xy).long()
    z_index = ((z_position + h_cm/2) / pixel_size_z).long()

    # Valid mask
    valid = (
        (x_index >= 0) & (x_index < num_pixels_xy) &
        (y_index >= 0) & (y_index < num_pixels_xy) &
        (z_index >= 0) & (z_index < num_pixels_z)
    )

    jj, n = torch.nonzero(valid, as_tuple=True)

    trajectory_idx = torch.stack([
        n,
        x_index[jj, n],
        y_index[jj, n],
        z_index[jj, n]
    ], dim=1)

    return trajectory_idx


def find_hull_intersection(
    hull: torch.Tensor,
    trajectory_idx: torch.Tensor,
    direction: str,
    N: int,
    l_cm: float,
    num_pixels_xy: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find entry/exit distances for hull

    Args:
        hull: Hull mask tensor (boolean)
        trajectory_idx: Trajectory indices [proton_id, x, y, z]
        direction: 'in' or 'out'
        N: Number of protons
        l_cm: Length in cm
        num_pixels_xy: Number of pixels in xy
        device: PyTorch device

    Returns:
        Tuple of (d_s, d_s_index, indices_interaction)
    """
    pixel_size_xy = l_cm / num_pixels_xy

    # Handle empty trajectory case
    if len(trajectory_idx) == 0:
        d_s = torch.full((N,), torch.inf, device=device)
        d_s_index = torch.full((N,), -1 if direction == 'out' else num_pixels_xy + 1,
                              device=device, dtype=torch.long)
        indices_interaction = torch.zeros(N, dtype=torch.bool, device=device)
        return d_s, d_s_index, indices_interaction

    p_idx = trajectory_idx[:, 0]
    x_idx = trajectory_idx[:, 1]
    y_idx = trajectory_idx[:, 2]
    z_idx = trajectory_idx[:, 3]

    # Keep only voxels inside hull
    inside = hull[x_idx, y_idx, z_idx]
    p_idx = p_idx[inside]
    x_idx = x_idx[inside]

    indices_interaction = torch.zeros(N, dtype=torch.bool, device=device)
    indices_interaction[p_idx] = True

    if direction == "in":
        x_init = torch.full((N,), num_pixels_xy + 1, device=device, dtype=torch.long)
        x_init.scatter_reduce_(0, p_idx, x_idx, reduce="amin")
        d_s_index = x_init

        d_s = torch.full((N,), torch.inf, device=device)
        valid = indices_interaction
        d_s[valid] = (l_cm / 2 + index_to_position(d_s_index[valid].float(), l_cm, pixel_size_xy))

    elif direction == "out":
        x_init = torch.full((N,), -1, device=device, dtype=torch.long)
        x_init.scatter_reduce_(0, p_idx, x_idx, reduce="amax")
        d_s_index = x_init

        d_s = torch.full((N,), torch.inf, device=device)
        valid = indices_interaction
        d_s[valid] = (l_cm / 2 - index_to_position(d_s_index[valid].float(), l_cm, pixel_size_xy))

    else:
        raise ValueError("direction must be 'in' or 'out'")

    return d_s, d_s_index, indices_interaction


def compute_mlp_img_recon_style(
    input_data_gpu: Dict[str, torch.Tensor],
    mlp_params_gpu: Dict[str, torch.Tensor],
    hull: Optional[torch.Tensor],
    l_mm: float,
    d_mm: float,
    h_mm: float,
    num_pixels_xy: int,
    num_pixels_z: int,
    device: torch.device,
    return_debug_images: bool = False
) -> Tuple:
    """
    Compute MLP using img_recon.py formula (simple and fast)

    Args:
        input_data_gpu: Dict with positions_u0-u3, positions_v0-v3, WEPL
        mlp_params_gpu: Dict with X_position, G1, G2, H1, H2
        hull: Optional hull mask for filtering
        l_mm: Length in mm
        d_mm: Detector spacing in mm
        h_mm: Height in mm
        num_pixels_xy: Number of pixels in xy
        num_pixels_z: Number of pixels in z
        device: PyTorch device
        return_debug_images: If True, return separate inside/outside hull images

    Returns:
        If return_debug_images=False:
            Tuple of (WEPL_projection_img, count_img)
        If return_debug_images=True:
            Tuple of (WEPL_inside, count_inside, WEPL_outside, count_outside)
    """
    pixel_size_xy = l_mm / num_pixels_xy
    pixel_size_z = h_mm / num_pixels_z
    shift_u = l_mm / 2
    shift_v = h_mm / 2

    # Extract positions (img_recon.py format)
    positions_u0 = input_data_gpu["positions_u0"]  # cm
    positions_u2 = input_data_gpu["positions_u2"]  # cm
    positions_u3 = input_data_gpu["positions_u3"]  # cm
    positions_v0 = input_data_gpu["positions_v0"]  # cm
    positions_v2 = input_data_gpu["positions_v2"]  # cm
    positions_v3 = input_data_gpu["positions_v3"]  # cm
    WEPL = input_data_gpu["WEPL"]  # cm

    # Convert mm to cm
    l_cm = l_mm / 10
    d_cm = d_mm / 10

    # MLP parameters
    X_position_gpu = mlp_params_gpu["X_position"] / 10  # mm to cm
    G1_gpu = mlp_params_gpu["G1"]
    G2_gpu = mlp_params_gpu["G2"]
    H1_gpu = mlp_params_gpu["H1"]
    H2_gpu = mlp_params_gpu["H2"]

    num_protons = positions_u0.shape[0]
    num_x_positions = X_position_gpu.shape[0]

    # Compute angles (img_recon.py formula)
    theta_u = torch.atan((positions_u3 - positions_u2) / d_cm).unsqueeze(1)
    theta_v = torch.atan((positions_v3 - positions_v2) / d_cm).unsqueeze(1)
    U = (positions_u2 - positions_u0).unsqueeze(1)
    V = (positions_v2 - positions_v0).unsqueeze(1)

    # Broadcast MLP parameters
    X_position_gpu = X_position_gpu.unsqueeze(0)  # (1, num_pixels)
    G1_G2 = (G1_gpu + G2_gpu).unsqueeze(0)
    H1_H2 = (H1_gpu + H2_gpu).unsqueeze(0)

    # Compute MLP positions (img_recon.py formula)
    Y_position = (theta_u * G1_G2 +
                 (U - (l_cm - X_position_gpu) * torch.sin(theta_u)) * H1_H2 +
                 positions_u0.unsqueeze(1) + shift_u / 10)  # cm

    Z_position = (theta_v * G1_G2 +
                 (V - (l_cm - X_position_gpu) * torch.sin(theta_v)) * H1_H2 +
                 positions_v0.unsqueeze(1) + shift_v / 10)  # cm

    # Convert to indices
    X_index = torch.floor(X_position_gpu / (pixel_size_xy / 10)).long().expand(num_protons, -1)
    Y_index = torch.floor(Y_position / (pixel_size_xy / 10)).long()
    Z_index = torch.floor(Z_position / (pixel_size_z / 10)).long()

    # Boundary mask
    mask_bounds = ((Z_index >= 0) & (Z_index < num_pixels_z) &
                   (Y_index >= 0) & (Y_index < num_pixels_xy) &
                   (X_index >= 0) & (X_index < num_pixels_xy))

    # If hull provided and debug mode requested
    if hull is not None and return_debug_images:
        # Get hull mask for valid boundary points
        X_valid = X_index[mask_bounds]
        Y_valid = Y_index[mask_bounds]
        Z_valid = Z_index[mask_bounds]

        inside_hull = hull[X_valid, Y_valid, Z_valid]

        # Create full hull mask
        mask_flat = mask_bounds.view(-1)
        inside_hull_full = torch.zeros_like(mask_flat, dtype=torch.bool)
        inside_hull_full[mask_flat] = inside_hull
        mask_hull = inside_hull_full.view(mask_bounds.shape)

        # Separate inside and outside hull
        mask_inside = mask_bounds & mask_hull
        mask_outside = mask_bounds & (~mask_hull)

        # Initialize images
        WEPL_inside = torch.zeros((num_pixels_xy, num_pixels_xy, num_pixels_z),
                                 device=device, dtype=torch.float32)
        count_inside = torch.zeros((num_pixels_xy, num_pixels_xy, num_pixels_z),
                                  device=device, dtype=torch.float32)
        WEPL_outside = torch.zeros((num_pixels_xy, num_pixels_xy, num_pixels_z),
                                  device=device, dtype=torch.float32)
        count_outside = torch.zeros((num_pixels_xy, num_pixels_xy, num_pixels_z),
                                   device=device, dtype=torch.float32)

        # Scatter inside hull
        valid_Y_in = Y_index[mask_inside]
        valid_X_in = X_index[mask_inside]
        valid_Z_in = Z_index[mask_inside]
        valid_WEPL_in = WEPL.unsqueeze(1).expand(-1, num_x_positions)[mask_inside]
        linear_indices_in = (valid_X_in * num_pixels_xy * num_pixels_z +
                            valid_Y_in * num_pixels_z + valid_Z_in)
        WEPL_inside.view(-1).scatter_add_(0, linear_indices_in, valid_WEPL_in)
        count_inside.view(-1).scatter_add_(0, linear_indices_in, torch.ones_like(valid_WEPL_in))

        # Scatter outside hull
        valid_Y_out = Y_index[mask_outside]
        valid_X_out = X_index[mask_outside]
        valid_Z_out = Z_index[mask_outside]
        valid_WEPL_out = WEPL.unsqueeze(1).expand(-1, num_x_positions)[mask_outside]
        linear_indices_out = (valid_X_out * num_pixels_xy * num_pixels_z +
                             valid_Y_out * num_pixels_z + valid_Z_out)
        WEPL_outside.view(-1).scatter_add_(0, linear_indices_out, valid_WEPL_out)
        count_outside.view(-1).scatter_add_(0, linear_indices_out, torch.ones_like(valid_WEPL_out))

        return WEPL_inside, count_inside, WEPL_outside, count_outside

    # Normal mode: apply hull mask if provided
    mask = mask_bounds
    if hull is not None:
        X_valid = X_index[mask_bounds]
        Y_valid = Y_index[mask_bounds]
        Z_valid = Z_index[mask_bounds]

        inside_hull = hull[X_valid, Y_valid, Z_valid]

        mask_flat = mask_bounds.view(-1)
        inside_hull_full = torch.zeros_like(mask_flat, dtype=torch.bool)
        inside_hull_full[mask_flat] = inside_hull
        mask = inside_hull_full.view(mask_bounds.shape)

    valid_Y = Y_index[mask]
    valid_X = X_index[mask]
    valid_Z = Z_index[mask]
    valid_WEPL = WEPL.unsqueeze(1).expand(-1, num_x_positions)[mask]

    # Initialize projection images
    WEPL_projection_img = torch.zeros((num_pixels_xy, num_pixels_xy, num_pixels_z),
                                      device=device, dtype=torch.float32)
    count_img = torch.zeros((num_pixels_xy, num_pixels_xy, num_pixels_z),
                           device=device, dtype=torch.float32)

    # Scatter add
    linear_indices = (valid_X * num_pixels_xy * num_pixels_z +
                     valid_Y * num_pixels_z + valid_Z)

    WEPL_projection_img.view(-1).scatter_add_(0, linear_indices, valid_WEPL)
    count_img.view(-1).scatter_add_(0, linear_indices, torch.ones_like(valid_WEPL))

    return WEPL_projection_img, count_img


def compute_mlp_vectorized_ultra_fast(
    P0_batch: torch.Tensor,
    P2_batch: torch.Tensor,
    d_s_in_batch: torch.Tensor,
    d_s_out_batch: torch.Tensor,
    start_index_batch: torch.Tensor,
    end_index_batch: torch.Tensor,
    mlp_params_cache_gpu: Dict[str, torch.Tensor],
    trajectory_idx0: torch.Tensor,
    trajectory_idx2: torch.Tensor,
    WEPL_batch: torch.Tensor,
    hull: torch.Tensor,
    l_cm: float,
    h_cm: float,
    num_pixels_xy: int,
    num_pixels_z: int,
    device: torch.device,
    use_mixed_precision: bool = True,
    profiling_dict: Optional[Dict] = None
) -> Optional[Dict[str, torch.Tensor]]:
    """
    ULTRA-FAST: Pure GPU scatter operations - NO LOOPS!

    Returns complete voxel indices + WEPL values ready for scatter_add

    Args:
        P0_batch: Entry positions/angles [N, 4] (y, z, angle_y, angle_z)
        P2_batch: Exit positions/angles [N, 4]
        d_s_in_batch: Entry distances [N]
        d_s_out_batch: Exit distances [N]
        start_index_batch: Start indices in hull [N]
        end_index_batch: End indices in hull [N]
        mlp_params_cache_gpu: Cached MLP parameters on GPU
        trajectory_idx0: Entry trajectory indices
        trajectory_idx2: Exit trajectory indices
        WEPL_batch: WEPL values [N]
        hull: Hull mask tensor
        l_cm: Length in cm
        h_cm: Height in cm
        num_pixels_xy: Number of pixels in xy
        num_pixels_z: Number of pixels in z
        device: PyTorch device
        use_mixed_precision: Use FP16 for speed
        profiling_dict: Optional profiling dictionary

    Returns:
        Dict with keys: x_idx, y_idx, z_idx, wepl (or None if no valid protons)
    """
    pixel_size_xy = l_cm / num_pixels_xy
    pixel_size_z = h_cm / num_pixels_z

    # Filter valid protons (those that interact with hull)
    valid_mask = (d_s_in_batch != torch.inf) & (d_s_out_batch != torch.inf)

    # Profiling
    if profiling_dict is not None:
        profiling_dict['with_hull'] += valid_mask.sum().item()
        profiling_dict['without_hull'] += (~valid_mask).sum().item()

    Sigma1_gpu = mlp_params_cache_gpu["Sigma1"]
    Sigma2_gpu = mlp_params_cache_gpu["Sigma2"]
    R0_gpu = mlp_params_cache_gpu["R0"]
    R1_gpu = mlp_params_cache_gpu["R1"]

    # Initialize output tensors
    all_x_indices = []
    all_y_indices = []
    all_z_indices = []
    all_wepl_values = []

    N_total = len(P0_batch)

    # ===================================================================
    # PART 1: Handle protons WITH hull interaction (use MLP)
    # ===================================================================
    if valid_mask.any():
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        P0_valid = P0_batch[valid_mask]
        P2_valid = P2_batch[valid_mask]
        d_s_in_valid = d_s_in_batch[valid_mask]
        d_s_out_valid = d_s_out_batch[valid_mask]
        start_index_valid = start_index_batch[valid_mask].long()
        end_index_valid = end_index_batch[valid_mask].long()
        WEPL_valid = WEPL_batch[valid_mask]

        M = P0_valid.shape[0]

        # Bounds check
        valid_bounds = (start_index_valid >= 0) & (end_index_valid < num_pixels_xy) & (start_index_valid <= end_index_valid)

        if valid_bounds.any():
            P0_valid = P0_valid[valid_bounds]
            P2_valid = P2_valid[valid_bounds]
            d_s_in_valid = d_s_in_valid[valid_bounds]
            d_s_out_valid = d_s_out_valid[valid_bounds]
            start_index_valid = start_index_valid[valid_bounds]
            end_index_valid = end_index_valid[valid_bounds]
            WEPL_valid = WEPL_valid[valid_bounds]
            valid_indices = valid_indices[valid_bounds]
            M = P0_valid.shape[0]

            # Create expanded pixel indices
            num_pixels_per_proton = end_index_valid - start_index_valid + 1
            max_pixels = num_pixels_per_proton.max().item()

            batch_indices = torch.arange(M, device=device).unsqueeze(1).expand(-1, max_pixels)
            pixel_offsets = torch.arange(max_pixels, device=device).unsqueeze(0).expand(M, -1)
            x_indices = start_index_valid.unsqueeze(1) + pixel_offsets
            valid_pixel_mask = pixel_offsets < num_pixels_per_proton.unsqueeze(1)
            end_indices_expanded = end_index_valid.unsqueeze(1).expand(-1, max_pixels)

            # Flatten
            valid_flat = valid_pixel_mask.flatten()
            batch_flat = batch_indices.flatten()[valid_flat]
            x_flat = x_indices.flatten()[valid_flat]
            end_flat = end_indices_expanded.flatten()[valid_flat]

            # Get MLP parameters (vectorized)
            Sigma1_y = Sigma1_gpu[x_flat, end_flat]
            Sigma2_y = Sigma2_gpu[x_flat, end_flat]
            R0_y = R0_gpu[x_flat, end_flat]
            R1_y = R1_gpu[x_flat, end_flat]

            # Mixed precision computation
            compute_dtype = Sigma1_y.dtype

            # S matrices
            d_s_in_batch_exp = d_s_in_valid[batch_flat].to(compute_dtype)
            d_s_out_batch_exp = d_s_out_valid[batch_flat].to(compute_dtype)

            S_in = torch.eye(2, device=device, dtype=compute_dtype).unsqueeze(0).expand(len(batch_flat), -1, -1).clone()
            S_in[:, 0, 1] = d_s_in_batch_exp

            S_out = torch.eye(2, device=device, dtype=compute_dtype).unsqueeze(0).expand(len(batch_flat), -1, -1).clone()
            S_out[:, 0, 1] = d_s_out_batch_exp

            # MLP for Y direction
            P0_y = P0_valid[batch_flat][:, [0, 2]].unsqueeze(-1).to(compute_dtype)
            P2_y = P2_valid[batch_flat][:, [0, 2]].unsqueeze(-1).to(compute_dtype)

            P0_y_t = S_in @ P0_y

            # FP16 workaround: linalg.inv/solve requires FP32
            if compute_dtype == torch.float16:
                S_out_inv = torch.linalg.inv(S_out.float()).half()
                P2_y_t = S_out_inv @ P2_y

                eps = 1e-4
                eye = torch.eye(2, device=device, dtype=compute_dtype).unsqueeze(0)
                Sigma1_y_reg = Sigma1_y + eps * eye
                Sigma2_y_reg = Sigma2_y + eps * eye

                inv_Sigma1_y = torch.linalg.inv(Sigma1_y_reg.float()).half()
                inv_Sigma2_y = torch.linalg.inv(Sigma2_y_reg.float()).half()
                R1t_y = R1_y.transpose(-1, -2)

                A_y = inv_Sigma1_y + R1t_y @ inv_Sigma2_y @ R1_y
                B_y = inv_Sigma1_y @ (R0_y @ P0_y_t) + R1t_y @ inv_Sigma2_y @ P2_y_t

                P_MLP_y = torch.linalg.solve(A_y.float(), B_y.float()).half()
            else:
                S_out_inv = torch.linalg.inv(S_out)
                P2_y_t = S_out_inv @ P2_y

                eps = 1e-8
                eye = torch.eye(2, device=device, dtype=compute_dtype).unsqueeze(0)
                Sigma1_y_reg = Sigma1_y + eps * eye
                Sigma2_y_reg = Sigma2_y + eps * eye

                inv_Sigma1_y = torch.linalg.inv(Sigma1_y_reg)
                inv_Sigma2_y = torch.linalg.inv(Sigma2_y_reg)
                R1t_y = R1_y.transpose(-1, -2)

                A_y = inv_Sigma1_y + R1t_y @ inv_Sigma2_y @ R1_y
                B_y = inv_Sigma1_y @ (R0_y @ P0_y_t) + R1t_y @ inv_Sigma2_y @ P2_y_t

                P_MLP_y = torch.linalg.solve(A_y, B_y)

            y_mlp = P_MLP_y[:, 0, 0].float()
            y_idx_mlp = ((y_mlp + l_cm/2) / pixel_size_xy).long()

            # MLP for Z direction
            P0_z = P0_valid[batch_flat][:, [1, 3]].unsqueeze(-1).to(compute_dtype)
            P2_z = P2_valid[batch_flat][:, [1, 3]].unsqueeze(-1).to(compute_dtype)

            P0_z_t = S_in @ P0_z
            P2_z_t = S_out_inv @ P2_z

            Sigma1_z = Sigma1_gpu[x_flat, end_flat]
            Sigma2_z = Sigma2_gpu[x_flat, end_flat]
            R0_z = R0_gpu[x_flat, end_flat]
            R1_z = R1_gpu[x_flat, end_flat]

            if compute_dtype == torch.float16:
                Sigma1_z_reg = Sigma1_z + eps * eye
                Sigma2_z_reg = Sigma2_z + eps * eye

                inv_Sigma1_z = torch.linalg.inv(Sigma1_z_reg.float()).half()
                inv_Sigma2_z = torch.linalg.inv(Sigma2_z_reg.float()).half()
                R1t_z = R1_z.transpose(-1, -2)

                A_z = inv_Sigma1_z + R1t_z @ inv_Sigma2_z @ R1_z
                B_z = inv_Sigma1_z @ (R0_z @ P0_z_t) + R1t_z @ inv_Sigma2_z @ P2_z_t

                P_MLP_z = torch.linalg.solve(A_z.float(), B_z.float()).half()
            else:
                Sigma1_z_reg = Sigma1_z + eps * eye
                Sigma2_z_reg = Sigma2_z + eps * eye

                inv_Sigma1_z = torch.linalg.inv(Sigma1_z_reg)
                inv_Sigma2_z = torch.linalg.inv(Sigma2_z_reg)
                R1t_z = R1_z.transpose(-1, -2)

                A_z = inv_Sigma1_z + R1t_z @ inv_Sigma2_z @ R1_z
                B_z = inv_Sigma1_z @ (R0_z @ P0_z_t) + R1t_z @ inv_Sigma2_z @ P2_z_t

                P_MLP_z = torch.linalg.solve(A_z, B_z)

            z_mlp = P_MLP_z[:, 0, 0].float()
            z_idx_mlp = ((z_mlp + h_cm/2) / pixel_size_z).long()

            # Bounds check for MLP results
            mlp_valid = (
                (x_flat >= 0) & (x_flat < num_pixels_xy) &
                (y_idx_mlp >= 0) & (y_idx_mlp < num_pixels_xy) &
                (z_idx_mlp >= 0) & (z_idx_mlp < num_pixels_z)
            )

            # Add MLP voxels
            all_x_indices.append(x_flat[mlp_valid])
            all_y_indices.append(y_idx_mlp[mlp_valid])
            all_z_indices.append(z_idx_mlp[mlp_valid])
            all_wepl_values.append(WEPL_valid[batch_flat[mlp_valid]].float())

            # VECTORIZED: Add straight segments BEFORE and AFTER MLP
            if len(trajectory_idx0) > 0:
                lookup_start = torch.full((N_total,), num_pixels_xy + 1, device=device, dtype=torch.long)
                lookup_start[valid_indices] = start_index_valid
                lookup_wepl_0 = torch.zeros((N_total,), device=device, dtype=torch.float32)
                lookup_wepl_0[valid_indices] = WEPL_valid.float()

                traj0_proton_ids = trajectory_idx0[:, 0]
                voxel_start_idx = lookup_start[traj0_proton_ids]
                voxel_x_idx = trajectory_idx0[:, 1]

                before_mask = (voxel_x_idx < voxel_start_idx) & (voxel_start_idx != num_pixels_xy + 1)

                if before_mask.any():
                    before_voxels = trajectory_idx0[before_mask]
                    all_x_indices.append(before_voxels[:, 1])
                    all_y_indices.append(before_voxels[:, 2])
                    all_z_indices.append(before_voxels[:, 3])
                    all_wepl_values.append(lookup_wepl_0[before_voxels[:, 0]])

            if len(trajectory_idx2) > 0:
                lookup_end = torch.full((N_total,), -1, device=device, dtype=torch.long)
                lookup_end[valid_indices] = end_index_valid
                lookup_wepl_2 = torch.zeros((N_total,), device=device, dtype=torch.float32)
                lookup_wepl_2[valid_indices] = WEPL_valid.float()

                traj2_proton_ids = trajectory_idx2[:, 0]
                voxel_end_idx = lookup_end[traj2_proton_ids]
                voxel_x_idx = trajectory_idx2[:, 1]

                after_mask = (voxel_x_idx > voxel_end_idx) & (voxel_end_idx != -1)

                if after_mask.any():
                    after_voxels = trajectory_idx2[after_mask]
                    all_x_indices.append(after_voxels[:, 1])
                    all_y_indices.append(after_voxels[:, 2])
                    all_z_indices.append(after_voxels[:, 3])
                    all_wepl_values.append(lookup_wepl_2[after_voxels[:, 0]])

    # ===================================================================
    # PART 2: Handle protons WITHOUT hull interaction (straight trajectory)
    # ===================================================================
    no_hull_mask = ~valid_mask
    if no_hull_mask.any():
        no_hull_indices = torch.nonzero(no_hull_mask, as_tuple=True)[0]

        lookup_wepl_no_hull = torch.zeros((N_total,), device=device, dtype=torch.float32)
        lookup_wepl_no_hull[no_hull_indices] = WEPL_batch[no_hull_mask]

        traj0_proton_ids = trajectory_idx0[:, 0]

        is_no_hull_proton = torch.zeros((N_total,), device=device, dtype=torch.bool)
        is_no_hull_proton[no_hull_indices] = True

        no_hull_voxel_mask = is_no_hull_proton[traj0_proton_ids]

        if no_hull_voxel_mask.any():
            no_hull_voxels = trajectory_idx0[no_hull_voxel_mask]

            all_x_indices.append(no_hull_voxels[:, 1])
            all_y_indices.append(no_hull_voxels[:, 2])
            all_z_indices.append(no_hull_voxels[:, 3])
            all_wepl_values.append(lookup_wepl_no_hull[no_hull_voxels[:, 0]])

    # ===================================================================
    # COMBINE ALL RESULTS
    # ===================================================================
    if len(all_x_indices) == 0:
        return None

    return {
        'x_idx': torch.cat(all_x_indices),
        'y_idx': torch.cat(all_y_indices),
        'z_idx': torch.cat(all_z_indices),
        'wepl': torch.cat(all_wepl_values)
    }
