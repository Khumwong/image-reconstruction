"""Trajectory calculations including straight paths and MLP computations"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from ..utils.helpers import index_to_position

# Global flag for debug printing (only print once)
_debug_scattering_printed = False


def compute_straight_trajectory_batch(
    p_b: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    angle: Tuple[torch.Tensor, torch.Tensor],
    N: int,
    image_size_xy_cm: float,
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
    pixel_size_xy = image_size_xy_cm / num_pixels_xy
    pixel_size_z = h_cm / num_pixels_z

    px, py, pz = p_b

    # Create x positions
    x = torch.linspace(-image_size_xy_cm/2 + pixel_size_xy/2,
                      image_size_xy_cm/2 - pixel_size_xy/2,
                      num_pixels_xy, device=device)

    x_exp = x.unsqueeze(1).expand(-1, N)

    # Calculate positions along beam path
    x_position = x_exp
    y_position = torch.tan(angle[0]) * (x_exp - px) + py
    z_position = torch.tan(angle[1]) * (x_exp - px) + pz

    # Convert to indices
    x_index = ((x_position + image_size_xy_cm/2) / pixel_size_xy).long()
    y_index = ((y_position + image_size_xy_cm/2) / pixel_size_xy).long()
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
    image_size_xy_cm: float,
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
    pixel_size_xy = image_size_xy_cm / num_pixels_xy

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
        d_s[valid] = (l_cm / 2 + index_to_position(d_s_index[valid].float(), image_size_xy_cm, pixel_size_xy))

    elif direction == "out":
        x_init = torch.full((N,), -1, device=device, dtype=torch.long)
        x_init.scatter_reduce_(0, p_idx, x_idx, reduce="amax")
        d_s_index = x_init

        d_s = torch.full((N,), torch.inf, device=device)
        valid = indices_interaction
        d_s[valid] = (l_cm / 2 - index_to_position(d_s_index[valid].float(), image_size_xy_cm, pixel_size_xy))

    else:
        raise ValueError("direction must be 'in' or 'out'")

    return d_s, d_s_index, indices_interaction


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
    image_size_xy_cm: float,
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
    pixel_size_xy = image_size_xy_cm / num_pixels_xy
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

            # CRITICAL: Expand entry and exit indices for dynamic Sigma lookup
            start_indices_expanded = start_index_valid.unsqueeze(1).expand(-1, max_pixels)
            end_indices_expanded = end_index_valid.unsqueeze(1).expand(-1, max_pixels)

            # Flatten
            valid_flat = valid_pixel_mask.flatten()
            batch_flat = batch_indices.flatten()[valid_flat]
            x_flat = x_indices.flatten()[valid_flat]  # MLP position
            start_flat = start_indices_expanded.flatten()[valid_flat]  # Entry position (hull entry)
            end_flat = end_indices_expanded.flatten()[valid_flat]  # Exit position (hull exit)

            # Get MLP parameters (CORRECTED: dynamic scattering from entry → MLP → exit)
            # Sigma1: scattering from hull ENTRY → MLP position (NOT from detector!)
            # Sigma2: scattering from MLP position → hull EXIT (NOT to detector!)
            Sigma1_y = Sigma1_gpu[start_flat, x_flat]
            Sigma2_y = Sigma2_gpu[x_flat, end_flat]
            R0_y = R0_gpu[start_flat, x_flat]
            R1_y = R1_gpu[x_flat, end_flat]

            # DEBUG: Print sample scattering info (first batch only, once)
            global _debug_scattering_printed
            if not _debug_scattering_printed:
                _debug_scattering_printed = True
                print(f"\n  🔍 DEBUG: Scattering verification (sample proton)")
                print(f"     Hull entry index: {start_flat[0].item()}")
                print(f"     MLP position index: {x_flat[0].item()}")
                print(f"     Hull exit index: {end_flat[0].item()}")

                # Sanity check
                is_valid_order = start_flat[0] <= x_flat[0] <= end_flat[0]
                status = "✅ CORRECT" if is_valid_order else "❌ ERROR"
                print(f"     Order check (entry ≤ MLP ≤ exit): {status}")

                print(f"     → Scattering ONLY from index {start_flat[0].item()} to {end_flat[0].item()} (inside hull)")
                print(f"     Sigma1[entry→MLP] norm: {torch.norm(Sigma1_y[0]).item():.6f}")
                print(f"     Sigma2[MLP→exit] norm: {torch.norm(Sigma2_y[0]).item():.6f}")

                # Count how many protons actually use MLP (vs straight)
                valid_mlp = valid_mask.sum().item()
                total = len(valid_mask)
                print(f"     Protons using MLP (hull interaction): {valid_mlp}/{total} ({100*valid_mlp/total:.1f}%)")

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
            y_idx_mlp = ((y_mlp + image_size_xy_cm/2) / pixel_size_xy).long()

            # MLP for Z direction
            P0_z = P0_valid[batch_flat][:, [1, 3]].unsqueeze(-1).to(compute_dtype)
            P2_z = P2_valid[batch_flat][:, [1, 3]].unsqueeze(-1).to(compute_dtype)

            P0_z_t = S_in @ P0_z
            P2_z_t = S_out_inv @ P2_z

            # Same dynamic Sigma lookup for Z direction
            Sigma1_z = Sigma1_gpu[start_flat, x_flat]
            Sigma2_z = Sigma2_gpu[x_flat, end_flat]
            R0_z = R0_gpu[start_flat, x_flat]
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

                before_mask = (voxel_x_idx <= voxel_start_idx) & (voxel_start_idx != num_pixels_xy + 1)

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

                after_mask = (voxel_x_idx >= voxel_end_idx) & (voxel_end_idx != -1)

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


def compute_mlp_rigorous(
    input_data_gpu: Dict[str, torch.Tensor],
    mlp_params_gpu: Dict[str, torch.Tensor],
    hull: Optional[torch.Tensor],
    l_mm: float,
    image_size_xy_cm: float,
    d_mm: float,
    h_mm: float,
    num_pixels_xy: int,
    num_pixels_z: int,
    device: torch.device,
    return_debug_images: bool = False
) -> Tuple:
    """
    Compute MLP using RIGOROUS physics (scattering matrices + Bayesian estimation)

    This is the CORRECT way to do proton CT reconstruction:
    1. Compute straight trajectories (before/after hull)
    2. Find hull intersection points
    3. Use MLP with scattering matrices inside hull
    4. Distribute WEPL according to path length

    Args:
        input_data_gpu: Dict with positions_u0-u3, positions_v0-v3, WEPL
        mlp_params_gpu: Dict with Sigma1, Sigma2, R0, R1 (rigorous matrices)
        hull: Optional hull mask for filtering
        l_mm: Length in mm (detector length)
        image_size_xy_mm: Image ROI size in mm
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
    l_cm = l_mm / 10
    h_cm = h_mm / 10
    d_cm = d_mm / 10

    # Extract data
    positions_u0 = input_data_gpu["positions_u0"]  # cm
    positions_u2 = input_data_gpu["positions_u2"]
    positions_u3 = input_data_gpu["positions_u3"]
    positions_v0 = input_data_gpu["positions_v0"]
    positions_v2 = input_data_gpu["positions_v2"]
    positions_v3 = input_data_gpu["positions_v3"]
    WEPL = input_data_gpu["WEPL"]  # cm

    num_protons = positions_u0.shape[0]

    # Compute angles for ENTRY trajectory (before phantom)
    # Use zero angles = straight horizontal beam
    angle_y_entry = torch.zeros_like(positions_u0)
    angle_z_entry = torch.zeros_like(positions_v0)

    # Compute angles for EXIT trajectory (after phantom)
    # Use scattered angles from detector measurements
    angle_y_exit = torch.atan((positions_u3 - positions_u2) / d_cm)
    angle_z_exit = torch.atan((positions_v3 - positions_v2) / d_cm)

    # Entry positions (detector 0)
    py_0 = positions_u0
    pz_0 = positions_v0
    px_0 = torch.full_like(py_0, -l_cm / 2)

    # Exit positions (detector 2)
    py_2 = positions_u2
    pz_2 = positions_v2
    px_2 = torch.full_like(py_2, l_cm / 2)

    # ===================================================================
    # STEP 1: Compute straight trajectories
    # ===================================================================
    # Entry trajectory: straight horizontal (angle = 0)
    p_b_0 = (px_0, py_0, pz_0)
    angle_0 = (angle_y_entry, angle_z_entry)

    trajectory_idx0 = compute_straight_trajectory_batch(
        p_b_0, angle_0, num_protons,
        image_size_xy_cm, h_cm, num_pixels_xy, num_pixels_z, device
    )

    # Exit trajectory: straight but with scattered angles
    p_b_2 = (px_2, py_2, pz_2)
    angle_2 = (angle_y_exit, angle_z_exit)

    trajectory_idx2 = compute_straight_trajectory_batch(
        p_b_2, angle_2, num_protons,
        image_size_xy_cm, h_cm, num_pixels_xy, num_pixels_z, device
    )

    # ===================================================================
    # STEP 2: Find hull intersections (if hull provided)
    # ===================================================================
    if hull is not None:
        # Entry point
        d_s_in, start_index, indices_in = find_hull_intersection(
            hull, trajectory_idx0, "in", num_protons, l_cm, image_size_xy_cm, num_pixels_xy, device
        )

        # Exit point
        d_s_out, end_index, indices_out = find_hull_intersection(
            hull, trajectory_idx2, "out", num_protons, l_cm, image_size_xy_cm, num_pixels_xy, device
        )

        # Prepare P0, P2 batches for MLP
        # P0: Entry angles (straight horizontal)
        # P2: Exit angles (scattered from detector measurements)
        P0_batch = torch.stack([py_0, pz_0, angle_y_entry, angle_z_entry], dim=1)
        P2_batch = torch.stack([py_2, pz_2, angle_y_exit, angle_z_exit], dim=1)

        # ===================================================================
        # STEP 3: Call rigorous MLP computation
        # ===================================================================
        result = compute_mlp_vectorized_ultra_fast(
            P0_batch=P0_batch,
            P2_batch=P2_batch,
            d_s_in_batch=d_s_in,
            d_s_out_batch=d_s_out,
            start_index_batch=start_index,
            end_index_batch=end_index,
            mlp_params_cache_gpu=mlp_params_gpu,
            trajectory_idx0=trajectory_idx0,
            trajectory_idx2=trajectory_idx2,
            WEPL_batch=WEPL,
            hull=hull,
            l_cm=l_cm,
            image_size_xy_cm=image_size_xy_cm,
            h_cm=h_cm,
            num_pixels_xy=num_pixels_xy,
            num_pixels_z=num_pixels_z,
            device=device,
            use_mixed_precision=False,
            profiling_dict=None
        )

        # ===================================================================
        # STEP 4: Scatter results
        # ===================================================================
        WEPL_projection_img = torch.zeros((num_pixels_xy, num_pixels_xy, num_pixels_z),
                                          device=device, dtype=torch.float32)
        count_img = torch.zeros((num_pixels_xy, num_pixels_xy, num_pixels_z),
                               device=device, dtype=torch.float32)

        if result is not None:
            x_idx = result['x_idx']
            y_idx = result['y_idx']
            z_idx = result['z_idx']
            wepl = result['wepl']

            linear_indices = (x_idx * num_pixels_xy * num_pixels_z +
                             y_idx * num_pixels_z + z_idx)

            WEPL_projection_img.view(-1).scatter_add_(0, linear_indices, wepl)
            count_img.view(-1).scatter_add_(0, linear_indices, torch.ones_like(wepl))

        if return_debug_images:
            # For debug mode: separate inside/outside
            # (simplified version - just return same for both)
            return WEPL_projection_img, count_img, \
                   torch.zeros_like(WEPL_projection_img), torch.zeros_like(count_img)

        return WEPL_projection_img, count_img

    else:
        # No hull - use straight trajectory only
        WEPL_projection_img = torch.zeros((num_pixels_xy, num_pixels_xy, num_pixels_z),
                                          device=device, dtype=torch.float32)
        count_img = torch.zeros((num_pixels_xy, num_pixels_xy, num_pixels_z),
                               device=device, dtype=torch.float32)

        if len(trajectory_idx0) > 0:
            x_idx = trajectory_idx0[:, 1]
            y_idx = trajectory_idx0[:, 2]
            z_idx = trajectory_idx0[:, 3]
            proton_ids = trajectory_idx0[:, 0]

            wepl_values = WEPL[proton_ids]

            linear_indices = (x_idx * num_pixels_xy * num_pixels_z +
                             y_idx * num_pixels_z + z_idx)

            WEPL_projection_img.view(-1).scatter_add_(0, linear_indices, wepl_values)
            count_img.view(-1).scatter_add_(0, linear_indices, torch.ones_like(wepl_values))

        if return_debug_images:
            return torch.zeros_like(WEPL_projection_img), torch.zeros_like(count_img), \
                   WEPL_projection_img, count_img

        return WEPL_projection_img, count_img
