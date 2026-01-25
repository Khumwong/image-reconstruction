import numpy as np
import pandas as pd
import torch
import pickle

from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional, Union
from collections import defaultdict
from scipy.ndimage import rotate

import matplotlib.pyplot as plt
import torch.nn.functional as F

import os
import re
import time
import math

torch.cuda.empty_cache()


class Timer:
    """Simple timer context manager for profiling"""
    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"  ⏱️  {self.name}: {self.elapsed:.3f}s")


class HullImageReconstructionFastOptimized:
    """
    ULTRA-FAST GPU-accelerated proton CT reconstruction with rigorous physics.

    Key optimizations:
    - Pure GPU vectorization (NO loops over protons)
    - Batch processing with scatter operations
    - Rigorous MLP physics (Sigma matrices from hull_img_recon.py)
    - Input format from img_recon.py (results_run*_degree_X.csv)
    - Output structure from img_recon.py (proton_paths_2, etc.)
    """

    def __init__(self,
                 csv_paths: Iterable[Union[str, Path]],
                 output_path: Path,
                 Ein_MeV: float = 100.0,
                 init_angle: int = 0,
                 d_mm: float = 25.0,
                 l_mm: float = 111.125 - 12.5 + 16,
                 imageSizeZ_mm: float = 115.0,
                 numOfPixelImage_xy: int = 512,
                 numOfPixelImage_z: int = 512,
                 device: str = 'cuda'):

        # GPU device setup
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {total_mem:.2f} GB")
            torch.cuda.empty_cache()

        # Path validation
        self.paths: List[Path] = []
        for p in csv_paths:
            pp = Path(p)
            if not pp.is_file():
                raise FileNotFoundError(f"CSV not found: {pp}")
            if pp.suffix.lower() != ".csv":
                raise ValueError(f"Not a .csv file: {pp}")
            self.paths.append(pp)
        self.output_path = Path(output_path)

        # Geometry parameters
        self.d_mm = float(d_mm)
        self.l_mm = float(l_mm)
        self.l_cm = float(l_mm) / 10
        self.imageSizeZ_mm = float(imageSizeZ_mm)
        self.h_cm = float(imageSizeZ_mm) / 10

        # Energy
        self.Ein_MeV = float(Ein_MeV)

        # Image parameters
        self.numOfPixelImage_xy = int(numOfPixelImage_xy)
        self.numOfPixelImage_z = int(numOfPixelImage_z)
        self.init_angle = init_angle

        # Cache for MLP parameters on GPU (avoid repeated transfers)
        self._mlp_params_cache_gpu = None

        # Enable mixed precision for faster computation on Tensor Cores
        self.use_mixed_precision = True
        if self.device.type == 'cuda':
            print(f"🚀 Mixed Precision (FP16) enabled for Tensor Cores")

    @staticmethod
    def _extract_angle_from_filename(path: Path) -> Optional[int]:
        """
        Extract angle directly from filename.
        Pattern: results_run506113706_251213113712_degree_90.csv
        Returns: angle (int) or None if not found
        """
        _PAT = re.compile(r"results_run\d+_\d+_degree_(\d+)\.csv$", re.IGNORECASE)
        m = _PAT.search(path.name)
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _load_csv_fast(path: Path) -> pd.DataFrame:
        """
        OPTIMIZED: Load CSV with float32 from start.
        Expected columns: CentroidX0, CentroidY0, CentroidX1, CentroidY1,
                         predicted_residue_energy, track_len, offset_mm
        """
        df = pd.read_csv(
            path,
            usecols=["CentroidX0", "CentroidY0", "CentroidX1", "CentroidY1",
                    "predicted_residue_energy", "track_len", "offset_mm"],
            dtype={"CentroidX0": "float32", "CentroidY0": "float32",
                   "CentroidX1": "float32", "CentroidY1": "float32",
                   "predicted_residue_energy": "float32",
                   "track_len": "int32",
                   "offset_mm": "float32"},
            engine="c", memory_map=True
        )

        # Filter out single-hit tracks and rename columns
        df = df[df["track_len"] != 1].rename(columns={
            "CentroidX0": "PosX0",
            "CentroidY0": "PosY0",
            "CentroidX1": "PosX1",
            "CentroidY1": "PosY1",
            "predicted_residue_energy": "E"
        }).drop(columns=["track_len"])

        return df

    @staticmethod
    def _positionToIndex(position, l, pixelSize):
        return (position + l/2) / pixelSize

    @staticmethod
    def _indexToPosition(index, l, pixelSize):
        return (index - l/2) * pixelSize

    def fClean_data_batch(self, df_dict: Dict[float, pd.DataFrame]) -> Dict[float, Dict[str, torch.Tensor]]:
        """OPTIMIZED: Batch process all offset groups at once"""

        def RangeEnergyModel(E_mev: np.ndarray) -> np.ndarray:
            A = 2.2e-3
            B = 1.77
            return 10.0 * A * np.power(E_mev, B)

        result = {}

        for offset, df in df_dict.items():
            center_offset = (offset - 99) / 10  # Convert to cm
            num_protons = len(df)

            # All operations in float32
            u0_np = np.full(num_protons, -center_offset, dtype=np.float32)
            u2_np = (df["PosX0"].values / 10 - center_offset).astype(np.float32)
            u3_np = (df["PosX1"].values / 10 - center_offset).astype(np.float32)
            v0_np = np.zeros(num_protons, dtype=np.float32)
            v2_np = (df["PosY0"].values / 10).astype(np.float32)
            v3_np = (df["PosY1"].values / 10).astype(np.float32)
            wepl_np = RangeEnergyModel(df["E"].values).astype(np.float32)

            # Single GPU transfer per offset
            result[offset] = {
                "positions_u0": torch.from_numpy(u0_np).to(self.device),
                "positions_u1": torch.from_numpy(u0_np).to(self.device),
                "positions_u2": torch.from_numpy(u2_np).to(self.device),
                "positions_u3": torch.from_numpy(u3_np).to(self.device),
                "positions_v0": torch.from_numpy(v0_np).to(self.device),
                "positions_v1": torch.from_numpy(v0_np).to(self.device),
                "positions_v2": torch.from_numpy(v2_np).to(self.device),
                "positions_v3": torch.from_numpy(v3_np).to(self.device),
                "WEPL": torch.from_numpy(wepl_np).to(self.device)  # GPU now!
            }

        return result

    def fGet_MLP_parameters_rigorous(self, recalculate=False, use_fast=True):
        """Rigorous MLP with full scattering physics (hull_img_recon.py style)

        Args:
            recalculate: Force recomputation
            use_fast: Use fast analytical integration (1000x faster)
        """
        device = self.device
        l = self.l_cm
        numOfPixel = self.numOfPixelImage_xy

        fileName = f"MLP_parameters_rigorous_lMax_{l}_numOfPixelMax_{numOfPixel}.pkl"

        # Load if exists
        if os.path.exists(fileName) and not recalculate:
            with open(fileName, "rb") as file:
                MLP_parameters = pickle.load(file)
            print(f"[LOAD] Rigorous MLP Parameters loaded from: {fileName}")
            return MLP_parameters

        print(f"[CALC] Computing rigorous MLP parameters for: {fileName}")
        if use_fast:
            print("  Using FAST analytical integration (1000x speedup)")
        else:
            print("  Using numerical integration (scipy.quad)")

        # Physics calculation
        def invertOfBeta_sq_p_sq_coeffs():
            """Return polynomial coefficients for invertOfBeta_sq_p_sq"""
            return np.array([7.457e-6, 4.548e-7, -5.777e-8, 1.301e-8, -9.228e-10, 2.687e-11])

        def analytical_sigma_sq_n(xi, xf, n):
            """FAST analytical integration - 1000x faster than scipy.quad

            Computes: ∫[xi to xf] (xf - x)^n × Σ(aₖ·x^k) dx analytically
            """
            coeffs = invertOfBeta_sq_p_sq_coeffs()
            E0 = 13.6
            X0 = 36.1

            if xf <= xi:
                return 0.0

            factor = (1 + 0.038*np.log((xf-xi) / X0))**2

            # Analytical integration using binomial expansion
            integral = 0.0
            for k, ak in enumerate(coeffs):
                # Expand (xf - x)^n * x^k and integrate term by term
                for j in range(k + 1):
                    # Binomial coefficient C(k,j)
                    binom_coeff = math.comb(k, j)
                    # Term: C(k,j) × xf^(k-j) × (-1)^j × (xf-x)^(n+j)
                    coeff = ak * binom_coeff * (xf ** (k - j)) * ((-1) ** j)
                    # Integrate (xf-x)^(n+j) from xi to xf
                    # Result: [(xf-xi)^(n+j+1)] / (n+j+1)
                    integral += coeff * ((xf - xi) ** (n + j + 1)) / (n + j + 1)

            return E0**2 * factor * integral / X0

        def numerical_sigma_sq_n(xi, xf, n):
            """Original slow scipy.quad integration"""
            from scipy.integrate import quad

            def invertOfBeta_sq_p_sq(x):
                coeffs = invertOfBeta_sq_p_sq_coeffs()
                return sum(coeffs[i] * x**i for i in range(len(coeffs)))

            def integrand(x):
                return (xf - x)**n * invertOfBeta_sq_p_sq(x)

            E0 = 13.6
            X0 = 36.1
            factor = (1 + 0.038*np.log((xf-xi) / X0))**2
            return (E0**2 * factor * quad(integrand, xi, xf)[0] / X0)

        # Choose integration method
        sigma_sq_n = analytical_sigma_sq_n if use_fast else numerical_sigma_sq_n

        def Sigma(xi, xf):
            return np.array([
                [sigma_sq_n(xi, xf, 2), sigma_sq_n(xi, xf, 1)],
                [sigma_sq_n(xi, xf, 1), sigma_sq_n(xi, xf, 0)]
            ], dtype=np.float32)

        def R(xi, xf):
            return np.array([
                [1, (xf - xi)],
                [0, 1]
            ], dtype=np.float32)

        # Pixel sampling
        x0 = 0
        x2 = l
        pixelSize = l / numOfPixel

        X1 = np.linspace(pixelSize/2, x2-pixelSize/2, numOfPixel, dtype=np.float32)
        Sigma1 = np.zeros((numOfPixel, numOfPixel, 2, 2), dtype=np.float32)
        Sigma2 = np.zeros((numOfPixel, numOfPixel, 2, 2), dtype=np.float32)
        R0 = np.zeros((numOfPixel, numOfPixel, 2, 2), dtype=np.float32)
        R1 = np.zeros((numOfPixel, numOfPixel, 2, 2), dtype=np.float32)

        print(f"  Computing {numOfPixel}×{numOfPixel} MLP integrals...", end='', flush=True)
        t_start = time.time()

        # Compute all pairs (can be parallelized, but sequential is fast enough with analytical)
        for idx1, x1 in enumerate(X1):
            for idx2, x2 in enumerate(X1):
                if x2 > x1:
                    Sigma1[idx1, idx2] = Sigma(x0, x1)
                    Sigma2[idx1, idx2] = Sigma(x1, x2)
                    R0[idx1, idx2] = R(x0, x1)
                    R1[idx1, idx2] = R(x1, x2)

        elapsed = time.time() - t_start
        print(f" Done in {elapsed:.2f}s!")

        MLP_parameters = {
            "Sigma1": Sigma1,
            "Sigma2": Sigma2,
            "R0": R0,
            "R1": R1,
            "X1": X1
        }

        # Save
        with open(fileName, "wb") as file:
            pickle.dump(MLP_parameters, file)
        print(f"[SAVE] Parameters written to: {fileName}")

        return MLP_parameters

    def fMake_fakeHull(self, geometry='oneFourth_cylinder'):
        """Create fake hull for testing"""
        device = self.device
        numOfPixel_y = self.numOfPixelImage_xy
        numOfPixel_x = self.numOfPixelImage_xy
        numOfPixel_z = self.numOfPixelImage_z

        l = self.l_cm
        h = self.h_cm

        y = np.linspace(-l/2, l/2, numOfPixel_y)
        x = np.linspace(-l/2, l/2, numOfPixel_x)
        z = np.linspace(-h/2, h/2, numOfPixel_z)

        X, Y, Z = np.meshgrid(x, y, z, indexing='xy')

        hull = np.full((numOfPixel_x, numOfPixel_y, numOfPixel_z), False, dtype=bool)

        if geometry == 'oneFourth_cylinder':
            R = 5  # radius in cm
            mask_quarter = (X <= 0) & (Y >= 0)
            mask_cylinder = (X**2 + Y**2) <= R**2
            hull[mask_quarter & mask_cylinder] = True
            return torch.from_numpy(hull).to(device)

        elif geometry == 'full_cylinder':
            R = 5
            mask_cylinder = (X**2 + Y**2) <= R**2
            hull[mask_cylinder] = True
            return torch.from_numpy(hull).to(device)

        else:
            raise ValueError(f"Unknown geometry type: {geometry}")

    def fStraight_trajectory_batch(self, p_b, angle, N):
        """GPU-accelerated straight trajectory calculation (batched)"""
        device = self.device
        l = self.l_cm
        h = self.h_cm
        numOfPixel_xy = self.numOfPixelImage_xy
        numOfPixel_z = self.numOfPixelImage_z

        pixelSize_xy = l / numOfPixel_xy
        pixelSize_z = h / numOfPixel_z

        px, py, pz = p_b

        # Create x positions
        x = torch.linspace(-l/2 + pixelSize_xy/2,
                          l/2 - pixelSize_xy/2,
                          numOfPixel_xy, device=device)

        x_exp = x.unsqueeze(1).expand(-1, N)

        # Calculate positions along beam path
        x_position = x_exp
        y_position = torch.tan(angle[0]) * (x_exp - px) + py
        z_position = torch.tan(angle[1]) * (x_exp - px) + pz

        # Convert to indices
        x_index = ((x_position + l/2) / pixelSize_xy).long()
        y_index = ((y_position + l/2) / pixelSize_xy).long()
        z_index = ((z_position + h/2) / pixelSize_z).long()

        # Valid mask
        valid = (
            (x_index >= 0) & (x_index < numOfPixel_xy) &
            (y_index >= 0) & (y_index < numOfPixel_xy) &
            (z_index >= 0) & (z_index < numOfPixel_z)
        )

        jj, n = torch.nonzero(valid, as_tuple=True)

        trajectory_idx = torch.stack([
            n,
            x_index[jj, n],
            y_index[jj, n],
            z_index[jj, n]
        ], dim=1)

        return trajectory_idx

    def fFind_d_s(self, hull, trajectory_idx, direction: str, N: int):
        """Find entry/exit distances for hull

        Args:
            hull: Hull mask tensor
            trajectory_idx: Trajectory indices [proton_id, x, y, z]
            direction: 'in' or 'out'
            N: Number of protons
        """
        device = self.device
        l = self.l_cm
        numOfPixel_xy = self.numOfPixelImage_xy
        pixelSize_xy = l / numOfPixel_xy

        # Handle empty trajectory case - return properly sized tensors
        if len(trajectory_idx) == 0:
            d_s = torch.full((N,), torch.inf, device=device)
            d_s_index = torch.full((N,), -1 if direction == 'out' else numOfPixel_xy + 1,
                                  device=device, dtype=torch.long)
            indices_Interaction = torch.zeros(N, dtype=torch.bool, device=device)
            return d_s, d_s_index, indices_Interaction

        p_idx = trajectory_idx[:, 0]
        x_idx = trajectory_idx[:, 1]
        y_idx = trajectory_idx[:, 2]
        z_idx = trajectory_idx[:, 3]

        # Keep only voxels inside hull
        inside = hull[x_idx, y_idx, z_idx]
        p_idx = p_idx[inside]
        x_idx = x_idx[inside]

        indices_Interaction = torch.zeros(N, dtype=torch.bool, device=device)
        indices_Interaction[p_idx] = True

        if direction == "in":
            x_init = torch.full((N,), numOfPixel_xy + 1, device=device, dtype=torch.long)
            x_init.scatter_reduce_(0, p_idx, x_idx, reduce="amin")
            d_s_index = x_init

            d_s = torch.full((N,), torch.inf, device=device)
            valid = indices_Interaction
            d_s[valid] = (l / 2 + self._indexToPosition(d_s_index[valid], l, pixelSize_xy))

        elif direction == "out":
            x_init = torch.full((N,), -1, device=device, dtype=torch.long)
            x_init.scatter_reduce_(0, p_idx, x_idx, reduce="amax")
            d_s_index = x_init

            d_s = torch.full((N,), torch.inf, device=device)
            valid = indices_Interaction
            d_s[valid] = (l / 2 - self._indexToPosition(d_s_index[valid], l, pixelSize_xy))

        else:
            raise ValueError("direction must be 'in' or 'out'")

        return d_s, d_s_index, indices_Interaction

    def fMLP_vectorized_ultra_fast(self, P0_batch, P2_batch, d_s_in_batch, d_s_out_batch,
                                    startIndex_batch, endIndex_batch, MLP_parameters,
                                    trajectory_idx0, trajectory_idx2, WEPL_batch,
                                    profiling_dict=None):
        """
        ULTRA-FAST: Pure GPU scatter operations - NO LOOPS!

        Returns complete voxel indices + WEPL values ready for scatter_add
        """
        device = self.device
        l = self.l_cm
        h = self.h_cm
        numOfPixel_xy = self.numOfPixelImage_xy
        numOfPixel_z = self.numOfPixelImage_z
        pixelSize_xy = l / numOfPixel_xy
        pixelSize_z = h / numOfPixel_z

        # Filter valid protons (those that interact with hull)
        valid_mask = (d_s_in_batch != torch.inf) & (d_s_out_batch != torch.inf)

        # ===== PROFILING: Count protons =====
        if profiling_dict is not None:
            profiling_dict['with_hull'] += valid_mask.sum().item()
            profiling_dict['without_hull'] += (~valid_mask).sum().item()

        # ===== OPTIMIZATION: Cache MLP parameters on GPU (avoid repeated CPU→GPU transfers) =====
        if self._mlp_params_cache_gpu is None:
            # Load to GPU and optionally convert to FP16 for Tensor Cores
            dtype = torch.float16 if self.use_mixed_precision else torch.float32
            self._mlp_params_cache_gpu = {
                "Sigma1": torch.from_numpy(MLP_parameters["Sigma1"]).to(device, dtype=dtype),
                "Sigma2": torch.from_numpy(MLP_parameters["Sigma2"]).to(device, dtype=dtype),
                "R0": torch.from_numpy(MLP_parameters["R0"]).to(device, dtype=dtype),
                "R1": torch.from_numpy(MLP_parameters["R1"]).to(device, dtype=dtype)
            }

        Sigma1_gpu = self._mlp_params_cache_gpu["Sigma1"]
        Sigma2_gpu = self._mlp_params_cache_gpu["Sigma2"]
        R0_gpu = self._mlp_params_cache_gpu["R0"]
        R1_gpu = self._mlp_params_cache_gpu["R1"]

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
            startIndex_valid = startIndex_batch[valid_mask].long()
            endIndex_valid = endIndex_batch[valid_mask].long()
            WEPL_valid = WEPL_batch[valid_mask]

            M = P0_valid.shape[0]

            # Bounds check
            valid_bounds = (startIndex_valid >= 0) & (endIndex_valid < numOfPixel_xy) & (startIndex_valid <= endIndex_valid)

            if valid_bounds.any():
                P0_valid = P0_valid[valid_bounds]
                P2_valid = P2_valid[valid_bounds]
                d_s_in_valid = d_s_in_valid[valid_bounds]
                d_s_out_valid = d_s_out_valid[valid_bounds]
                startIndex_valid = startIndex_valid[valid_bounds]
                endIndex_valid = endIndex_valid[valid_bounds]
                WEPL_valid = WEPL_valid[valid_bounds]
                valid_indices = valid_indices[valid_bounds]
                M = P0_valid.shape[0]

                # Create expanded pixel indices
                num_pixels_per_proton = endIndex_valid - startIndex_valid + 1
                max_pixels = num_pixels_per_proton.max().item()

                batch_indices = torch.arange(M, device=device).unsqueeze(1).expand(-1, max_pixels)
                pixel_offsets = torch.arange(max_pixels, device=device).unsqueeze(0).expand(M, -1)
                x_indices = startIndex_valid.unsqueeze(1) + pixel_offsets
                valid_pixel_mask = pixel_offsets < num_pixels_per_proton.unsqueeze(1)
                end_indices_expanded = endIndex_valid.unsqueeze(1).expand(-1, max_pixels)

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

                # ===== MIXED PRECISION: Use autocast for matrix operations =====
                # Convert inputs to match MLP parameter dtype (FP16 or FP32)
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

                # FP16 workaround: linalg.inv/solve requires FP32, compute in FP32 then convert back
                if compute_dtype == torch.float16:
                    # Do inversions in FP32 for numerical stability
                    S_out_inv = torch.linalg.inv(S_out.float()).half()
                    P2_y_t = S_out_inv @ P2_y

                    # Add small regularization to avoid singular matrices
                    eps = 1e-4
                    eye = torch.eye(2, device=device, dtype=compute_dtype).unsqueeze(0)
                    Sigma1_y_reg = Sigma1_y + eps * eye
                    Sigma2_y_reg = Sigma2_y + eps * eye

                    # Inversions in FP32
                    inv_Sigma1_y = torch.linalg.inv(Sigma1_y_reg.float()).half()
                    inv_Sigma2_y = torch.linalg.inv(Sigma2_y_reg.float()).half()
                    R1t_y = R1_y.transpose(-1, -2)

                    A_y = inv_Sigma1_y + R1t_y @ inv_Sigma2_y @ R1_y
                    B_y = inv_Sigma1_y @ (R0_y @ P0_y_t) + R1t_y @ inv_Sigma2_y @ P2_y_t

                    # Solve in FP32 for stability
                    P_MLP_y = torch.linalg.solve(A_y.float(), B_y.float()).half()
                else:
                    # FP32 path (original)
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

                y_mlp = P_MLP_y[:, 0, 0].float()  # Convert back to FP32 for indexing
                y_idx_mlp = ((y_mlp + l/2) / pixelSize_xy).long()

                # MLP for Z direction
                P0_z = P0_valid[batch_flat][:, [1, 3]].unsqueeze(-1).to(compute_dtype)
                P2_z = P2_valid[batch_flat][:, [1, 3]].unsqueeze(-1).to(compute_dtype)

                P0_z_t = S_in @ P0_z
                P2_z_t = S_out_inv @ P2_z

                Sigma1_z = Sigma1_gpu[x_flat, end_flat]
                Sigma2_z = Sigma2_gpu[x_flat, end_flat]
                R0_z = R0_gpu[x_flat, end_flat]
                R1_z = R1_gpu[x_flat, end_flat]

                # Add regularization for Z direction
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

                z_mlp = P_MLP_z[:, 0, 0].float()  # Convert back to FP32 for indexing
                z_idx_mlp = ((z_mlp + h/2) / pixelSize_z).long()

                # Bounds check for MLP results
                mlp_valid = (
                    (x_flat >= 0) & (x_flat < numOfPixel_xy) &
                    (y_idx_mlp >= 0) & (y_idx_mlp < numOfPixel_xy) &
                    (z_idx_mlp >= 0) & (z_idx_mlp < numOfPixel_z)
                )

                # Now combine: [straight_before] + [MLP] + [straight_after] using GPU operations
                # For each valid proton, we need:
                # - trajectory_idx0 before startIndex
                # - MLP voxels
                # - trajectory_idx2 after endIndex

                # Add MLP voxels (convert WEPL back to FP32 for accumulation)
                all_x_indices.append(x_flat[mlp_valid])
                all_y_indices.append(y_idx_mlp[mlp_valid])
                all_z_indices.append(z_idx_mlp[mlp_valid])
                all_wepl_values.append(WEPL_valid[batch_flat[mlp_valid]].float())

                # VECTORIZED: Add straight segments BEFORE and AFTER MLP (NO PYTHON LOOPS!)
                # Process trajectory0 (before hull)
                if len(trajectory_idx0) > 0:
                    # Create lookup tables for startIndex and WEPL (scatter valid values into full array)
                    lookup_start = torch.full((N_total,), numOfPixel_xy + 1, device=device, dtype=torch.long)
                    lookup_start[valid_indices] = startIndex_valid
                    lookup_wepl_0 = torch.zeros((N_total,), device=device, dtype=torch.float32)
                    lookup_wepl_0[valid_indices] = WEPL_valid.float()

                    # Get start indices for each voxel in trajectory0
                    traj0_proton_ids = trajectory_idx0[:, 0]
                    voxel_start_idx = lookup_start[traj0_proton_ids]
                    voxel_x_idx = trajectory_idx0[:, 1]

                    # Keep only voxels BEFORE startIndex
                    before_mask = (voxel_x_idx < voxel_start_idx) & (voxel_start_idx != numOfPixel_xy + 1)

                    if before_mask.any():
                        before_voxels = trajectory_idx0[before_mask]
                        all_x_indices.append(before_voxels[:, 1])
                        all_y_indices.append(before_voxels[:, 2])
                        all_z_indices.append(before_voxels[:, 3])
                        all_wepl_values.append(lookup_wepl_0[before_voxels[:, 0]])

                # Process trajectory2 (after hull)
                if len(trajectory_idx2) > 0:
                    # Create lookup tables for endIndex and WEPL
                    lookup_end = torch.full((N_total,), -1, device=device, dtype=torch.long)
                    lookup_end[valid_indices] = endIndex_valid
                    lookup_wepl_2 = torch.zeros((N_total,), device=device, dtype=torch.float32)
                    lookup_wepl_2[valid_indices] = WEPL_valid.float()

                    # Get end indices for each voxel in trajectory2
                    traj2_proton_ids = trajectory_idx2[:, 0]
                    voxel_end_idx = lookup_end[traj2_proton_ids]
                    voxel_x_idx = trajectory_idx2[:, 1]

                    # Keep only voxels AFTER endIndex
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
            # ===== PROFILING: Time the no-hull section =====
            t_no_hull_start = time.time()

            # VECTORIZED VERSION - NO PYTHON LOOPS! 🚀
            # Create a lookup table mapping proton IDs to WEPL values
            no_hull_indices = torch.nonzero(no_hull_mask, as_tuple=True)[0]

            # Create full-size WEPL lookup (scatter no-hull WEPL values into full array)
            lookup_wepl_no_hull = torch.zeros((N_total,), device=device, dtype=torch.float32)
            lookup_wepl_no_hull[no_hull_indices] = WEPL_batch[no_hull_mask]

            # Filter trajectory_idx0 to keep only voxels from no-hull protons
            # trajectory_idx0[:, 0] contains proton IDs
            traj0_proton_ids = trajectory_idx0[:, 0]

            # Create a mask: which voxels belong to no-hull protons?
            # Use lookup table for fast checking
            is_no_hull_proton = torch.zeros((N_total,), device=device, dtype=torch.bool)
            is_no_hull_proton[no_hull_indices] = True

            # Apply mask to trajectory voxels
            no_hull_voxel_mask = is_no_hull_proton[traj0_proton_ids]

            if no_hull_voxel_mask.any():
                # Extract voxels that belong to no-hull protons (all vectorized!)
                no_hull_voxels = trajectory_idx0[no_hull_voxel_mask]

                # Get coordinates
                all_x_indices.append(no_hull_voxels[:, 1])
                all_y_indices.append(no_hull_voxels[:, 2])
                all_z_indices.append(no_hull_voxels[:, 3])

                # Get WEPL values using lookup table (vectorized!)
                all_wepl_values.append(lookup_wepl_no_hull[no_hull_voxels[:, 0]])

            # ===== PROFILING: Record time =====
            if profiling_dict is not None:
                profiling_dict['time_no_hull_loop'] += (time.time() - t_no_hull_start)

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

    def fBackProjection(self, use_hull=True, hull_geometry='oneFourth_cylinder') -> np.ndarray:
        """
        ULTRA-FAST GPU-accelerated back-projection.
        Output structure matches img_recon.py
        """
        from scipy.ndimage import rotate

        out_path = self.output_path
        # Match img_recon.py output structure
        os.makedirs(out_path / "proton_paths", exist_ok=True)
        os.makedirs(out_path / "proton_paths_2", exist_ok=True)
        os.makedirs(out_path / "proton_paths_images_2", exist_ok=True)

        def _save_image_fast(array: np.ndarray, filename: str):
            """Fast image save - EXACT match to img_recon.py format"""
            slice_sums = array.sum(axis=(0, 1))
            v_position = int(slice_sums.argmax())

            # Get slice and apply transformations: rotate 90° CW, then flip LR
            slice_data = array[:, :, v_position]

            # Check if this is hull image (boolean array) - skip transformation for hull
            if filename == "hull":
                transformed_slice = slice_data
                extent_horizontal = [-60, 60]  # Z axis
                extent_vertical = [39, 159]    # X axis
                xlabel = 'Z axis (mm)'
                ylabel = 'X axis (mm)'
            else:
                # Rotate 90° clockwise: rot90(k=-1) or equivalently rot90(k=3)
                slice_rotated = np.rot90(slice_data, k=-1)
                # Flip left-right
                transformed_slice = np.fliplr(slice_rotated)

                # After rotation and flip, axes swap and coordinates change
                extent_horizontal = [-60, 60]  # Now Y axis
                extent_vertical = [39, 159]    # Now X axis (reversed)
                xlabel = 'Y axis (mm)'
                ylabel = 'X axis (mm)'

            # Physical coordinates (EXACT match to img_recon.py)
            # Horizontal (Z): center at 0mm, range -60 to 60 mm
            # Vertical (X): pivot at 99mm, range 159 (top) to 39 (bottom) mm

            plt.figure(figsize=(10, 7))
            im = plt.imshow(transformed_slice, cmap='gray',
                           extent=[extent_horizontal[0], extent_horizontal[1],
                                   extent_vertical[0], extent_vertical[1]],
                           aspect='auto', origin='upper')

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f'Slice {v_position}/{self.numOfPixelImage_z}')

            # Add colorbar
            cbar = plt.colorbar(im, label='Intensity')

            # Crosshair at pivot point (coordinates adjust based on transformation)
            if filename == "hull":
                plt.axhline(y=99, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
                plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
            else:
                # After 90° CW rotation + LR flip: (x, y) → (-y, x)
                # Original pivot: (0, 99) → new pivot: (-99, 0)
                plt.axhline(y=99, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
                plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

            plt.savefig(out_path / "proton_paths_images_2" / f"{filename}.png",
                    bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close()

        def _save_debug_overlay_rotated_hull(array_inside: np.ndarray, array_outside: np.ndarray,
                                            filename: str, hull_mask: Optional[np.ndarray] = None):
            """Save debug overlay for images WITH rotated hull (count, WEPL, average at specific angle)

            These images show proton paths at a specific angle, so the hull is also rotated.
            Use origin='upper' to match the coordinate system.

            Args:
                array_inside: Inside hull proton data
                array_outside: Outside hull proton data
                filename: Output filename
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

            # Apply transformations: rotate 90° CW, then flip LR
            combined_slice_rotated = np.rot90(combined_slice, k=-1)
            combined_slice_transformed = np.fliplr(combined_slice_rotated)

            # Physical coordinates after 90° CW rotation + LR flip
            extent_horizontal = [-60, 60]    # Y axis
            extent_vertical = [39, 159]      # X axis

            plt.figure(figsize=(10, 8))

            # Show transformed proton paths - use origin='upper' for correct orientation
            im = plt.imshow(combined_slice_transformed, cmap='viridis',
                        extent=[extent_horizontal[0], extent_horizontal[1],
                                extent_vertical[0], extent_vertical[1]],
                        aspect='auto', origin='upper')

            plt.xlabel('Y axis (mm)')
            plt.ylabel('X axis (mm)')
            plt.title(f'Proton Paths (Rotated Hull)\nSlice {v_position}/{self.numOfPixelImage_z}')
            plt.colorbar(im, label='Intensity')

            # Draw phantom mask boundary as contour
            if hull_mask is not None:
                from skimage import measure

                # Get the hull slice at the same position
                hull_slice = hull_mask[:, :, v_position]

                if hull_slice.any():
                    # Apply SAME transformations to hull: rotate 90° CW, then flip LR
                    hull_slice_rotated = np.rot90(hull_slice, k=-1)
                    hull_slice_transformed = np.fliplr(hull_slice_rotated)

                    # Flip vertically to match origin='upper' coordinate system
                    hull_slice_transformed = np.flipud(hull_slice_transformed)

                    # Find contours of the phantom mask
                    contours = measure.find_contours(hull_slice_transformed, 0.5)

                    # Convert pixel indices to physical coordinates
                    pixel_size_xy = self.l_mm / self.numOfPixelImage_xy

                    # Draw all contours
                    for i, contour in enumerate(contours):
                        # contour is (N, 2) array with (row, col)
                        row_indices = contour[:, 0]
                        col_indices = contour[:, 1]

                        # Map to physical coordinates
                        y_mm = (col_indices + 14) * pixel_size_xy + extent_horizontal[0]
                        x_mm = (row_indices + 14) * pixel_size_xy + extent_vertical[0]

                        # Plot contour
                        if i == 0:
                            plt.plot(y_mm, x_mm, 'k-', linewidth=2, label='Phantom boundary')
                        else:
                            plt.plot(y_mm, x_mm, 'k-', linewidth=2)

                    plt.legend(loc='upper right')

            # Crosshair at pivot point
            plt.axhline(y=99, color='red', linestyle='--', linewidth=0.5, alpha=0.7, label='Center')
            plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.7, label='Pivot')

            plt.savefig(out_path / "proton_paths_images_2" / f"{filename}_debug_overlay.png",
                    bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()

        def _save_debug_overlay_back_rotated(array_inside: np.ndarray, array_outside: np.ndarray,
                                             filename: str, hull_mask: Optional[np.ndarray] = None):
            """Save debug overlay for back-rotated reconstruction (Re_img)

            These images are accumulated from all angles with back-rotation, so they show
            the final reconstruction. Hull should also be transformed to match.
            Use origin='upper' to match the coordinate system.

            Args:
                array_inside: Inside hull proton data (back-rotated)
                array_outside: Outside hull proton data (back-rotated)
                filename: Output filename
                hull_mask: Original hull mask (will be transformed to match data)
            """
            slice_sums_in = array_inside.sum(axis=(0, 1))
            slice_sums_out = array_outside.sum(axis=(0, 1))
            slice_sums_total = slice_sums_in + slice_sums_out
            v_position = int(slice_sums_total.argmax())

            # Get both slices and combine
            inside_slice = array_inside[:, :, v_position]
            outside_slice = array_outside[:, :, v_position]
            combined_slice = inside_slice + outside_slice

            # Apply transformations: rotate 90° CW, then flip LR
            combined_slice_rotated = np.rot90(combined_slice, k=-1)
            combined_slice_transformed = np.fliplr(combined_slice_rotated)

            # Physical coordinates after 90° CW rotation + LR flip
            extent_horizontal = [-60, 60]    # Y axis
            extent_vertical = [39, 159]      # X axis

            plt.figure(figsize=(10, 8))

            # Show transformed proton paths - use origin='upper' for correct orientation
            im = plt.imshow(combined_slice_transformed, cmap='viridis',
                        extent=[extent_horizontal[0], extent_horizontal[1],
                                extent_vertical[0], extent_vertical[1]],
                        aspect='auto', origin='upper')

            plt.xlabel('Y axis (mm)')
            plt.ylabel('X axis (mm)')
            plt.title(f'Proton Paths (Reconstructed)\nSlice {v_position}/{self.numOfPixelImage_z}')
            plt.colorbar(im, label='Intensity')

            # Draw phantom mask boundary as contour
            if hull_mask is not None:
                from skimage import measure

                # Get the hull slice at the same position
                hull_slice = hull_mask[:, :, v_position]

                if hull_slice.any():
                    # Apply SAME transformations to hull: rotate 90° CW, then flip LR
                    hull_slice_rotated = np.rot90(hull_slice, k=-1)
                    hull_slice_transformed = np.fliplr(hull_slice_rotated)

                    # Flip vertically to match origin='upper' coordinate system
                    hull_slice_transformed = np.flipud(hull_slice_transformed)

                    # Find contours of the phantom mask
                    contours = measure.find_contours(hull_slice_transformed, 0.5)

                    # Convert pixel indices to physical coordinates
                    pixel_size_xy = self.l_mm / self.numOfPixelImage_xy

                    # Draw all contours
                    for i, contour in enumerate(contours):
                        # contour is (N, 2) array with (row, col)
                        row_indices = contour[:, 0]
                        col_indices = contour[:, 1]

                        # Map to physical coordinates
                        y_mm = (col_indices + 14) * pixel_size_xy + extent_horizontal[0]
                        x_mm = (row_indices + 14) * pixel_size_xy + extent_vertical[0]

                        # Plot contour
                        if i == 0:
                            plt.plot(y_mm, x_mm, 'k-', linewidth=2, label='Phantom boundary')
                        else:
                            plt.plot(y_mm, x_mm, 'k-', linewidth=2)

                    plt.legend(loc='upper right')

            # Crosshair at pivot point
            plt.axhline(y=99, color='red', linestyle='--', linewidth=0.5, alpha=0.7, label='Center')
            plt.axvline(x=0, color='red', linestyle='--', linewidth=0.5, alpha=0.7, label='Pivot')

            plt.savefig(out_path / "proton_paths_images_2" / f"{filename}_debug_overlay.png",
                    bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()

        d = self.d_mm / 10  # Convert to cm
        numOfPixelImage_xy = self.numOfPixelImage_xy
        numOfPixelImage_z = self.numOfPixelImage_z
        l = self.l_cm
        h = self.h_cm

        print("\n" + "="*70)
        print("Computing rigorous MLP parameters...")
        print("="*70)
        t0 = time.time()
        MLP_parameters = self.fGet_MLP_parameters_rigorous(recalculate=False)
        print(f"  Done: {time.time()-t0:.1f}s\n")

        # Create hull if needed
        hull = None
        if use_hull:
            print(f"Creating hull: {hull_geometry}")
            hull = self.fMake_fakeHull(geometry=hull_geometry)
            _save_image_fast(hull.cpu().numpy(), filename="hull")

        # Group files by angle
        by_angle: Dict[int, List[Path]] = defaultdict(list)

        print("\nGrouping files by angle...")
        for path in self.paths:
            angle = self._extract_angle_from_filename(path)
            if angle is not None:
                by_angle[angle].append(path)
            else:
                raise ValueError(f"Cannot extract angle from filename: {path.name}\n"
                               f"Expected format: results_run*_*_degree_XX.csv")

        print(f"Found {len(by_angle)} angles: {sorted(by_angle.keys())}")

        Re_img_gpu = torch.zeros((numOfPixelImage_xy, numOfPixelImage_xy, numOfPixelImage_z),
                                device=self.device, dtype=torch.float32)

        # Debug reconstruction images (inside/outside hull tracking)
        Re_img_inside_gpu = torch.zeros((numOfPixelImage_xy, numOfPixelImage_xy, numOfPixelImage_z),
                                        device=self.device, dtype=torch.float32)
        Re_img_outside_gpu = torch.zeros((numOfPixelImage_xy, numOfPixelImage_xy, numOfPixelImage_z),
                                         device=self.device, dtype=torch.float32)

        for angle in sorted(by_angle.keys()):
            print("="*70)
            print(f"ANGLE {angle}°")
            print("="*70)
            t_angle = time.time()

            sum_WEPL_gpu = torch.zeros((numOfPixelImage_xy, numOfPixelImage_xy, numOfPixelImage_z),
                                      device=self.device, dtype=torch.float32)
            sum_count_gpu = torch.zeros((numOfPixelImage_xy, numOfPixelImage_xy, numOfPixelImage_z),
                                       device=self.device, dtype=torch.float32)

            # Debug images for visualization (inside/outside hull tracking)
            debug_WEPL_inside_gpu = torch.zeros((numOfPixelImage_xy, numOfPixelImage_xy, numOfPixelImage_z),
                                                device=self.device, dtype=torch.float32)
            debug_count_inside_gpu = torch.zeros((numOfPixelImage_xy, numOfPixelImage_xy, numOfPixelImage_z),
                                                 device=self.device, dtype=torch.float32)
            debug_WEPL_outside_gpu = torch.zeros((numOfPixelImage_xy, numOfPixelImage_xy, numOfPixelImage_z),
                                                 device=self.device, dtype=torch.float32)
            debug_count_outside_gpu = torch.zeros((numOfPixelImage_xy, numOfPixelImage_xy, numOfPixelImage_z),
                                                  device=self.device, dtype=torch.float32)

            # Rotate hull for this angle
            hull_rot = None
            if use_hull:
                print("  Rotating hull...", end='', flush=True)
                hull_np = hull.cpu().numpy()
                hull_rot_np = rotate(hull_np, angle= angle, axes=(0, 1), reshape=False, order=0)
                hull_rot = torch.from_numpy(hull_rot_np).to(self.device).bool()
                print(" Done!")

            # Process each file for this angle
            for file_idx, path in enumerate(by_angle[angle]):
                print(f"  File {file_idx+1}/{len(by_angle[angle])}: {path.name}")

                # Load CSV
                t_load = time.time()
                df_full = self._load_csv_fast(path)
                print(f"    Load: {time.time()-t_load:.2f}s", end='')

                # Split by offset
                unique_offsets = sorted(df_full['offset_mm'].unique())
                df_dict = {off: df_full[df_full['offset_mm'] == off] for off in unique_offsets}
                print(f" | Offsets: {len(unique_offsets)}", end='')

                # Batch process
                t_proc = time.time()
                input_data_dict = self.fClean_data_batch(df_dict)

                # Profiling dictionary
                profiling = {
                    'total_protons': 0,
                    'with_hull': 0,
                    'without_hull': 0,
                    'time_mlp_total': 0,
                    'time_no_hull_loop': 0,
                    'time_mlp_matrix': 0,
                    'time_mlp_scatter': 0
                }

                for offset, input_data_gpu in input_data_dict.items():
                    y0 = input_data_gpu["positions_u0"]
                    y2 = input_data_gpu["positions_u2"]
                    y3 = input_data_gpu["positions_u3"]
                    z0 = input_data_gpu["positions_v0"]
                    z2 = input_data_gpu["positions_v2"]
                    z3 = input_data_gpu["positions_v3"]
                    WEPL = input_data_gpu["WEPL"]

                    angle_proton_y0 = (input_data_gpu["positions_u1"] - y0) / d
                    angle_proton_y2 = (y3 - y2) / d
                    angle_proton_z0 = (input_data_gpu["positions_v1"] - z0) / d
                    angle_proton_z2 = (z3 - z2) / d

                    N = y0.shape[0]
                    profiling['total_protons'] += N
                    l_in = l_out = l / 2

                    p0 = torch.stack([torch.full((N,), -l_in, device=self.device), y0, z0])
                    p2 = torch.stack([torch.full((N,), l_out, device=self.device), y2, z2])
                    angle0 = torch.stack([angle_proton_y0, angle_proton_z0])
                    angle2 = torch.stack([angle_proton_y2, angle_proton_z2])

                    # Straight trajectories
                    trajectory_idx0 = self.fStraight_trajectory_batch(p0, angle0, N)
                    trajectory_idx2 = self.fStraight_trajectory_batch(p2, angle2, N)

                    if use_hull and hull_rot is not None:
                        # Find hull intersections
                        d_s_in, startIndex, _ = self.fFind_d_s(hull_rot, trajectory_idx0, direction='in', N=N)
                        d_s_out, endIndex, _ = self.fFind_d_s(hull_rot, trajectory_idx2, direction='out', N=N)

                        # Prepare batch data
                        P0_batch = torch.stack([p0[1], p0[2], angle0[0], angle0[1]], dim=1)
                        P2_batch = torch.stack([p2[1], p2[2], angle2[0], angle2[1]], dim=1)

                        # ===== PROFILING: Start MLP section =====
                        t_mlp_start = time.time()

                        # ULTRA-FAST MLP computation (normal reconstruction)
                        result = self.fMLP_vectorized_ultra_fast(
                            P0_batch, P2_batch, d_s_in, d_s_out,
                            startIndex, endIndex, MLP_parameters,
                            trajectory_idx0, trajectory_idx2, WEPL,
                            profiling_dict=profiling
                        )

                        profiling['time_mlp_total'] += (time.time() - t_mlp_start)

                        if result is not None:
                            # Pure scatter_add operations - NO LOOPS!
                            linear_indices = (result['x_idx'] * numOfPixelImage_xy * numOfPixelImage_z +
                                            result['y_idx'] * numOfPixelImage_z +
                                            result['z_idx'])

                            sum_WEPL_gpu.view(-1).scatter_add_(0, linear_indices, result['wepl'])
                            sum_count_gpu.view(-1).scatter_add_(0, linear_indices,
                                                               torch.ones_like(result['wepl']))

                            # Debug: Also accumulate all proton paths (inside + outside hull) for visualization
                            # Check which voxels are inside hull
                            inside_mask = hull_rot[result['x_idx'], result['y_idx'], result['z_idx']]
                            outside_mask = ~inside_mask

                            if inside_mask.any():
                                linear_indices_in = linear_indices[inside_mask]
                                wepl_in = result['wepl'][inside_mask]
                                debug_WEPL_inside_gpu.view(-1).scatter_add_(0, linear_indices_in, wepl_in)
                                debug_count_inside_gpu.view(-1).scatter_add_(0, linear_indices_in,
                                                                            torch.ones_like(wepl_in))

                            if outside_mask.any():
                                linear_indices_out = linear_indices[outside_mask]
                                wepl_out = result['wepl'][outside_mask]
                                debug_WEPL_outside_gpu.view(-1).scatter_add_(0, linear_indices_out, wepl_out)
                                debug_count_outside_gpu.view(-1).scatter_add_(0, linear_indices_out,
                                                                             torch.ones_like(wepl_out))
                    else:
                        # No hull - straight trajectory only
                        valid_x = trajectory_idx0[:, 1]
                        valid_y = trajectory_idx0[:, 2]
                        valid_z = trajectory_idx0[:, 3]
                        valid_proton_idx = trajectory_idx0[:, 0]

                        linear_indices = (valid_x * numOfPixelImage_xy * numOfPixelImage_z +
                                        valid_y * numOfPixelImage_z + valid_z)

                        wepl_expanded = WEPL[valid_proton_idx]
                        sum_WEPL_gpu.view(-1).scatter_add_(0, linear_indices, wepl_expanded)
                        sum_count_gpu.view(-1).scatter_add_(0, linear_indices,
                                                          torch.ones_like(wepl_expanded))

                print(f" | Process: {time.time()-t_proc:.2f}s")

                # ===== PROFILING OUTPUT =====
                print(f"    📊 Proton stats:")
                print(f"       Total: {profiling['total_protons']:,}")
                print(f"       With hull: {profiling['with_hull']:,} ({100*profiling['with_hull']/max(profiling['total_protons'],1):.1f}%)")
                print(f"       Without hull: {profiling['without_hull']:,} ({100*profiling['without_hull']/max(profiling['total_protons'],1):.1f}%)")
                print(f"    ⏱️  Timing breakdown:")
                total_compute = profiling['time_mlp_total'] + profiling['time_no_hull_loop']
                print(f"       MLP total: {profiling['time_mlp_total']:.2f}s ({100*profiling['time_mlp_total']/max(total_compute,0.001):.1f}%)")
                print(f"       No-hull loop: {profiling['time_no_hull_loop']:.2f}s ({100*profiling['time_no_hull_loop']/max(total_compute,0.001):.1f}%)")

                del df_full, df_dict, input_data_dict
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Save results - EXACT match to img_recon.py format
            sum_WEPL = sum_WEPL_gpu.cpu().numpy()
            sum_count = sum_count_gpu.cpu().numpy()

            # Save debug visualization (ONLY colored overlay images)
            debug_WEPL_inside = debug_WEPL_inside_gpu.cpu().numpy()
            debug_count_inside = debug_count_inside_gpu.cpu().numpy()
            debug_WEPL_outside = debug_WEPL_outside_gpu.cpu().numpy()
            debug_count_outside = debug_count_outside_gpu.cpu().numpy()

            print(f"    Saving debug overlay images...")
            hull_rotated_np_for_plot = hull_rot.cpu().numpy() if hull_rot is not None else None
            # For count and WEPL: use rotated hull version (hull is already rotated for this angle)
            _save_debug_overlay_rotated_hull(debug_count_inside, debug_count_outside,
                                             f"count_angle{angle}_degree", hull_rotated_np_for_plot)
            _save_debug_overlay_rotated_hull(debug_WEPL_inside, debug_WEPL_outside,
                                             f"WEPL_angle{angle}_degree", hull_rotated_np_for_plot)

            # Calculate average (inside and outside separately)
            A_inside_gpu = torch.where(debug_count_inside_gpu != 0,
                                      debug_WEPL_inside_gpu / debug_count_inside_gpu,
                                      torch.zeros_like(debug_WEPL_inside_gpu))
            A_outside_gpu = torch.where(debug_count_outside_gpu != 0,
                                       debug_WEPL_outside_gpu / debug_count_outside_gpu,
                                       torch.zeros_like(debug_WEPL_outside_gpu))

            A_inside = A_inside_gpu.cpu().numpy()
            A_outside = A_outside_gpu.cpu().numpy()

            # Combined average for normal reconstruction
            A_gpu = torch.where(sum_count_gpu != 0, sum_WEPL_gpu / sum_count_gpu,
                               torch.zeros_like(sum_WEPL_gpu))
            A = A_gpu.cpu().numpy()

            # Save to proton_paths folder (MATCH img_recon.py)
            np.save(f"{out_path}/proton_paths/average_angle{angle}_degree.npy", A_inside + A_outside)
            np.save(f"{out_path}/proton_paths/WEPL_angle{angle}_degree.npy", debug_WEPL_inside + debug_WEPL_outside)
            np.save(f"{out_path}/proton_paths/count_angle{angle}_degree.npy", debug_count_inside + debug_count_outside)

            # Save average debug overlay (hull is rotated for this angle)
            _save_debug_overlay_rotated_hull(A_inside, A_outside, f"average_angle{angle}_degree", hull_rotated_np_for_plot)

            # GPU rotation - large batches
            # IMPORTANT: negate angle for back-projection (reverse rotation)
            angle_rad = -angle * np.pi / 180.0
            theta = torch.tensor([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                [np.sin(angle_rad), np.cos(angle_rad), 0]],
                               dtype=torch.float32, device=self.device).unsqueeze(0)

            batch_size = 64

            # Rotate combined average (for normal reconstruction)
            A_rotated_gpu = torch.zeros_like(A_gpu)
            for batch_idx in range(0, numOfPixelImage_z, batch_size):
                end_z = min(batch_idx + batch_size, numOfPixelImage_z)
                b_size = end_z - batch_idx

                slices = A_gpu[:, :, batch_idx:end_z].permute(2, 0, 1).unsqueeze(1)
                grid = F.affine_grid(theta.expand(b_size, -1, -1), slices.size(), align_corners=False)
                rotated = F.grid_sample(slices, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                A_rotated_gpu[:, :, batch_idx:end_z] = rotated.squeeze(1).permute(1, 2, 0)

            Re_img_gpu += A_rotated_gpu

            # Rotate inside average (for debug)
            A_inside_rotated_gpu = torch.zeros_like(A_inside_gpu)
            for batch_idx in range(0, numOfPixelImage_z, batch_size):
                end_z = min(batch_idx + batch_size, numOfPixelImage_z)
                b_size = end_z - batch_idx

                slices = A_inside_gpu[:, :, batch_idx:end_z].permute(2, 0, 1).unsqueeze(1)
                grid = F.affine_grid(theta.expand(b_size, -1, -1), slices.size(), align_corners=False)
                rotated = F.grid_sample(slices, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                A_inside_rotated_gpu[:, :, batch_idx:end_z] = rotated.squeeze(1).permute(1, 2, 0)

            Re_img_inside_gpu += A_inside_rotated_gpu

            # Rotate outside average (for debug)
            A_outside_rotated_gpu = torch.zeros_like(A_outside_gpu)
            for batch_idx in range(0, numOfPixelImage_z, batch_size):
                end_z = min(batch_idx + batch_size, numOfPixelImage_z)
                b_size = end_z - batch_idx

                slices = A_outside_gpu[:, :, batch_idx:end_z].permute(2, 0, 1).unsqueeze(1)
                grid = F.affine_grid(theta.expand(b_size, -1, -1), slices.size(), align_corners=False)
                rotated = F.grid_sample(slices, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                A_outside_rotated_gpu[:, :, batch_idx:end_z] = rotated.squeeze(1).permute(1, 2, 0)

            Re_img_outside_gpu += A_outside_rotated_gpu

            # Save Re_img debug overlay (colored)
            Re_img_inside_intermediate = Re_img_inside_gpu.cpu().numpy()
            Re_img_outside_intermediate = Re_img_outside_gpu.cpu().numpy()
            # Use original hull (not rotated) for Re_img since it's back-rotated
            hull_original_np = hull.cpu().numpy() if hull is not None else None
            _save_debug_overlay_back_rotated(Re_img_inside_intermediate, Re_img_outside_intermediate,
                                            f"Re_img_angle{angle}_degree", hull_original_np)

            # Save Re_img grayscale (all paths: inside + outside)
            Re_img_all = Re_img_inside_intermediate + Re_img_outside_intermediate
            _save_image_fast(Re_img_all, f"Re_img_angle{angle}_degree")

            print(f"  ✅ Done: {time.time()-t_angle:.1f}s\n")

            del sum_WEPL_gpu, sum_count_gpu, A_gpu, A_rotated_gpu, sum_WEPL, sum_count, A, hull_rot
            del debug_WEPL_inside_gpu, debug_count_inside_gpu, debug_WEPL_outside_gpu, debug_count_outside_gpu
            del debug_WEPL_inside, debug_count_inside, debug_WEPL_outside, debug_count_outside
            del A_inside_gpu, A_outside_gpu, A_inside, A_outside
            del A_inside_rotated_gpu, A_outside_rotated_gpu
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return Re_img_gpu.cpu().numpy()


if __name__ == "__main__":
    import argparse

    time_start = time.time()

    parser = argparse.ArgumentParser(description="ULTRA-FAST hull-aware image reconstruction with rigorous physics")
    parser.add_argument("--csv_folder", type=str,
                       default="/home/sutpct/Workspace/test_pyeudaq_reader/ProcessedCSV",
                       help="Folder containing CSV files")
    parser.add_argument("--output", type=str, default="./output_hull_fast_optimized",
                       help="Output folder")
    parser.add_argument("--no_hull", action="store_true",
                       help="Disable hull geometry (default: use hull)")
    parser.add_argument("--hull_geometry", type=str, default="oneFourth_cylinder",
                       choices=["oneFourth_cylinder", "full_cylinder"],
                       help="Hull geometry type")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Computing device")
    parser.add_argument("--resolution_xy", type=int, default=512,
                       help="XY resolution (default: 512 to match img_recon.py)")
    parser.add_argument("--resolution_z", type=int, default=512,
                       help="Z resolution (default: 512 to match img_recon.py)")
    parser.add_argument("--init_angle", type=int, default=0,
                       help="Initial angle offset")

    args = parser.parse_args()

    processed_csv_folder = Path(args.csv_folder)
    csv_list = [f for f in os.listdir(processed_csv_folder) if f.endswith('.csv')]
    csv_paths = [os.path.join(processed_csv_folder, file) for file in csv_list]

    output_path = Path(args.output)
    os.makedirs(output_path, exist_ok=True)

    use_hull = not args.no_hull

    print(f"\n{'='*70}")
    print(f"ULTRA-FAST HULL-AWARE PROTON CT RECONSTRUCTION")
    print(f"{'='*70}")
    print(f"Files: {len(csv_paths)}")
    print(f"Resolution: {args.resolution_xy}×{args.resolution_xy}×{args.resolution_z}")
    print(f"Hull support: {use_hull}")
    print(f"Hull geometry: {args.hull_geometry}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    ir = HullImageReconstructionFastOptimized(
        csv_paths,
        output_path,
        Ein_MeV=100.0,
        init_angle=args.init_angle,
        device=args.device,
        numOfPixelImage_xy=args.resolution_xy,
        numOfPixelImage_z=args.resolution_z
    )

    R = ir.fBackProjection(use_hull=use_hull, hull_geometry=args.hull_geometry)
    np.save(output_path / "BackProjection_hull.npy", R)  # MATCH img_recon.py filename

    print(f"\n{'='*70}")
    print(f"✅ COMPLETE!")
    print(f"   Time: {time.time()-time_start:.1f}s ({(time.time()-time_start)/60:.1f} min)")
    print(f"{'='*70}\n")
