"""Main reconstruction class - orchestrates all modules"""
import os
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Dict, Optional
from collections import defaultdict

from .core import get_mlp_parameters_rigorous, create_hull, rotate_hull
from .processing import (
    load_csv_fast, clean_data_batch,
    compute_mlp_rigorous,
    rotate_image_gpu_batched, compute_average_image
)
from .visualization import (
    save_reimg_image, save_reimg_sum_image,
    save_average_debug_overlay, save_count_debug_overlay, save_wepl_debug_overlay,
    save_reimg_debug_overlay, save_reimg_sum_debug_overlay
)
from .utils import Timer, extract_angle_from_filename


class HullImageReconstruction:
    """
    ULTRA-FAST GPU-accelerated proton CT reconstruction with rigorous physics.

    This is a refactored modular version with improved maintainability.

    Key features:
    - Pure GPU vectorization (NO loops over protons)
    - Batch processing with scatter operations
    - Rigorous MLP physics (scattering matrices)
    - Modular architecture for easy modification
    """

    def __init__(self,
                 csv_paths: List[Union[str, Path]],
                 output_path: Union[str, Path],
                 Ein_MeV: float = 100.0,
                 init_angle: int = 0,
                 d_mm: float = 25.0,
                 l_mm: float = (111.125 - 12.5 + 16) * 2,
                 image_size_xy_mm: float = 120.0,
                 image_size_z_mm: float = 115.0,
                 num_pixels_xy: int = 512,
                 num_pixels_z: int = 512,
                 device: str = 'cuda',
                 radius_cm: float = 5.0,
                 batch_size: int = 64):
        """
        Initialize reconstruction engine

        Args:
            csv_paths: List of CSV file paths
            output_path: Output directory path
            Ein_MeV: Initial proton energy in MeV
            init_angle: Initial angle offset
            d_mm: Detector separation in mm
            l_mm: Object length in mm
            image_size_z_mm: Image size in Z direction (mm)
            num_pixels_xy: Number of pixels in XY plane
            num_pixels_z: Number of pixels in Z direction
            device: Computing device ('cuda' or 'cpu')
            radius_cm: Hull cylinder radius in cm
            batch_size: Batch size for image rotation
        """
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

        # Validate and store CSV paths
        self.paths: List[Path] = []
        for p in csv_paths:
            pp = Path(p)
            if not pp.is_file():
                raise FileNotFoundError(f"CSV not found: {pp}")
            if pp.suffix.lower() != ".csv":
                raise ValueError(f"Not a .csv file: {pp}")
            self.paths.append(pp)

        self.output_path = Path(output_path)

        # Geometry parameters (convert mm to cm)
        self.d_mm = float(d_mm)
        self.l_mm = float(l_mm)
        self.l_cm = float(l_mm) / 10
        self.image_size_z_mm = float(image_size_z_mm)
        self.h_mm = float(image_size_z_mm)
        self.h_cm = float(image_size_z_mm) / 10
        self.image_size_xy_mm = float(image_size_xy_mm)
        self.image_size_xy_cm = float(image_size_xy_mm) / 10

        # Energy
        self.Ein_MeV = float(Ein_MeV)

        # Image parameters
        self.num_pixels_xy = int(num_pixels_xy)
        self.num_pixels_z = int(num_pixels_z)
        self.init_angle = init_angle

        # Hull geometry
        self.radius_cm = float(radius_cm)

        # Processing parameters
        self.batch_size = int(batch_size)

        # Cache for MLP parameters on GPU (avoid repeated transfers)
        self._mlp_params_cache_gpu = None

    def reconstruct(
        self,
        use_hull: bool = True,
        hull_geometry: str = 'oneFourth_cylinder'
    ) -> np.ndarray:
        """
        Main reconstruction pipeline

        Args:
            use_hull: Whether to use hull geometry
            hull_geometry: Type of hull ('oneFourth_cylinder' or 'full_cylinder')

        Returns:
            3D reconstructed image as numpy array
        """
        # Create output directories
        os.makedirs(self.output_path / "proton_paths", exist_ok=True)
        os.makedirs(self.output_path / "proton_paths_2", exist_ok=True)
        # Create separate folders for each image type
        os.makedirs(self.output_path / "Re_img", exist_ok=True)
        os.makedirs(self.output_path / "Re_img_sum", exist_ok=True)
        os.makedirs(self.output_path / "count", exist_ok=True)
        os.makedirs(self.output_path / "WEPL", exist_ok=True)
        os.makedirs(self.output_path / "average", exist_ok=True)

        # Load MLP parameters (RIGOROUS version with scattering matrices)
        print("\n" + "="*70)
        print("Computing RIGOROUS MLP parameters (scattering matrices)...")
        print("="*70)
        t0 = time.time()
        mlp_parameters = get_mlp_parameters_rigorous(
            l_cm=self.l_cm,
            image_size_xy_cm=self.image_size_xy_cm,
            num_pixels=self.num_pixels_xy,
            recalculate=True,   # Force recompute with correct method
            use_fast=False      # Use numerical integration (CORRECT, but slow)
        )
        print(f"  Done: {time.time()-t0:.1f}s\n")

        # Create hull if needed
        hull = None
        if use_hull:
            print(f"Creating hull: {hull_geometry}")
            hull = create_hull(
                geometry=hull_geometry,
                num_pixels_xy=self.num_pixels_xy,
                num_pixels_z=self.num_pixels_z,
                image_size_xy_mm=self.image_size_xy_mm,
                h_cm=self.h_cm,
                device=self.device,
                radius_cm=self.radius_cm
            )
            save_reimg_image(
                hull.cpu().numpy(),
                "hull",
                self.output_path,
                self.num_pixels_z,
                self.image_size_xy_mm,
                self.num_pixels_xy
            )

        # Group files by angle
        by_angle = self._group_files_by_angle()
        print(f"Found {len(by_angle)} angles: {sorted(by_angle.keys())}")

        # Initialize reconstruction image
        re_img_gpu = torch.zeros(
            (self.num_pixels_xy, self.num_pixels_xy, self.num_pixels_z),
            device=self.device, dtype=torch.float32
        )

        # Debug images (inside/outside hull tracking)
        re_img_inside_gpu = torch.zeros_like(re_img_gpu)
        re_img_outside_gpu = torch.zeros_like(re_img_gpu)

        # Sum images (without averaging)
        re_img_sum_inside_gpu = torch.zeros_like(re_img_gpu)
        re_img_sum_outside_gpu = torch.zeros_like(re_img_gpu)

        # Process each angle
        for angle in sorted(by_angle.keys()):
            print("="*70)
            print(f"ANGLE {angle}°")
            print("="*70)
            t_angle = time.time()

            # Process this angle
            angle_result = self._process_angle(
                angle, by_angle[angle], hull, mlp_parameters
            )

            # Back-rotate and accumulate
            rotated_inside = rotate_image_gpu_batched(
                angle_result['inside_gpu'],
                -angle,  # Negative for back-projection
                self.num_pixels_z,
                self.device,
                self.batch_size
            )
            rotated_outside = rotate_image_gpu_batched(
                angle_result['outside_gpu'],
                -angle,
                self.num_pixels_z,
                self.device,
                self.batch_size
            )
            rotated_combined = rotate_image_gpu_batched(
                angle_result['combined_gpu'],
                -angle,
                self.num_pixels_z,
                self.device,
                self.batch_size
            )

            re_img_inside_gpu += rotated_inside
            re_img_outside_gpu += rotated_outside
            re_img_gpu += rotated_combined

            # Back-rotate and accumulate sum (without averaging)
            rotated_sum_inside = rotate_image_gpu_batched(
                angle_result['sum_inside_gpu'],
                -angle,
                self.num_pixels_z,
                self.device,
                self.batch_size
            )
            rotated_sum_outside = rotate_image_gpu_batched(
                angle_result['sum_outside_gpu'],
                -angle,
                self.num_pixels_z,
                self.device,
                self.batch_size
            )
            re_img_sum_inside_gpu += rotated_sum_inside
            re_img_sum_outside_gpu += rotated_sum_outside

            # Save intermediate results
            self._save_angle_results(angle, angle_result, hull)

            # Save Re_img debug overlay
            re_img_inside_intermediate = re_img_inside_gpu.cpu().numpy()
            re_img_outside_intermediate = re_img_outside_gpu.cpu().numpy()
            hull_original_np = hull.cpu().numpy() if hull is not None else None

            save_reimg_debug_overlay(
                re_img_inside_intermediate,
                re_img_outside_intermediate,
                f"Re_img_angle{angle}_degree",
                self.output_path,
                self.num_pixels_z,
                self.image_size_xy_mm,
                self.num_pixels_xy,
                hull_original_np
            )

            # Save Re_img grayscale
            re_img_all = re_img_inside_intermediate + re_img_outside_intermediate
            save_reimg_image(
                re_img_all,
                f"Re_img_angle{angle}_degree",
                self.output_path,
                self.num_pixels_z,
                self.image_size_xy_mm,
                self.num_pixels_xy
            )

            # Save Re_img_sum (sum without averaging)
            re_img_sum_inside_intermediate = re_img_sum_inside_gpu.cpu().numpy()
            re_img_sum_outside_intermediate = re_img_sum_outside_gpu.cpu().numpy()

            save_reimg_sum_debug_overlay(
                re_img_sum_inside_intermediate,
                re_img_sum_outside_intermediate,
                f"Re_img_sum_angle{angle}_degree",
                self.output_path,
                self.num_pixels_z,
                self.image_size_xy_mm,
                self.num_pixels_xy,
                hull_original_np
            )

            re_img_sum_all = re_img_sum_inside_intermediate + re_img_sum_outside_intermediate
            save_reimg_sum_image(
                re_img_sum_all,
                f"Re_img_sum_angle{angle}_degree",
                self.output_path,
                self.num_pixels_z,
                self.image_size_xy_mm,
                self.num_pixels_xy
            )

            print(f"  ✅ Done: {time.time()-t_angle:.1f}s\n")

            # Cleanup
            del angle_result, rotated_inside, rotated_outside, rotated_combined
            del rotated_sum_inside, rotated_sum_outside
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return re_img_gpu.cpu().numpy()

    def _group_files_by_angle(self) -> Dict[int, List[Path]]:
        """Group CSV files by angle"""
        by_angle = defaultdict(list)

        print("\nGrouping files by angle...")
        for path in self.paths:
            angle = extract_angle_from_filename(path)
            if angle is not None:
                by_angle[angle].append(path)
            else:
                raise ValueError(
                    f"Cannot extract angle from filename: {path.name}\n"
                    f"Expected format: results_run*_*_degree_XX.csv"
                )

        return by_angle

    def _process_angle(
        self,
        angle: int,
        paths: List[Path],
        hull: Optional[torch.Tensor],
        mlp_parameters: Dict
    ) -> Dict[str, torch.Tensor]:
        """Process all files for a given angle"""

        # Initialize accumulators
        sum_wepl_gpu = torch.zeros(
            (self.num_pixels_xy, self.num_pixels_xy, self.num_pixels_z),
            device=self.device, dtype=torch.float32
        )
        sum_count_gpu = torch.zeros_like(sum_wepl_gpu)

        debug_wepl_inside_gpu = torch.zeros_like(sum_wepl_gpu)
        debug_count_inside_gpu = torch.zeros_like(sum_wepl_gpu)
        debug_wepl_outside_gpu = torch.zeros_like(sum_wepl_gpu)
        debug_count_outside_gpu = torch.zeros_like(sum_wepl_gpu)

        # Rotate hull for this angle
        hull_rot = None
        if hull is not None:
            print("  Rotating hull...", end='', flush=True)
            hull_rot = rotate_hull(hull, angle, self.device)
            print(" Done!")

        # Initialize MLP cache on first use (RIGOROUS format: scattering matrices)
        if self._mlp_params_cache_gpu is None:
            self._mlp_params_cache_gpu = {
                "Sigma1": torch.from_numpy(mlp_parameters["Sigma1"]).to(self.device).float(),
                "Sigma2": torch.from_numpy(mlp_parameters["Sigma2"]).to(self.device).float(),
                "R0": torch.from_numpy(mlp_parameters["R0"]).to(self.device).float(),
                "R1": torch.from_numpy(mlp_parameters["R1"]).to(self.device).float(),
                "X1": torch.from_numpy(mlp_parameters["X1"]).to(self.device).float()
            }

        # Process each file
        for file_idx, path in enumerate(paths):
            print(f"  File {file_idx+1}/{len(paths)}: {path.name}")

            t_load = time.time()
            df_full = load_csv_fast(path)
            print(f"    Load: {time.time()-t_load:.2f}s", end='')

            # Split by offset
            unique_offsets = sorted(df_full['offset_mm'].unique())
            df_dict = {off: df_full[df_full['offset_mm'] == off] for off in unique_offsets}
            print(f" | Offsets: {len(unique_offsets)}", end='')

            # Batch process
            t_proc = time.time()
            input_data_dict = clean_data_batch(df_dict, self.device)

            profiling = {'total_protons': 0}

            for offset, input_data_gpu in input_data_dict.items():
                # Normal reconstruction (masked) - RIGOROUS MLP
                WEPL_img, count_img = compute_mlp_rigorous(
                    input_data_gpu, self._mlp_params_cache_gpu, hull_rot,
                    self.l_mm, self.image_size_xy_cm, self.d_mm, self.h_mm,
                    self.num_pixels_xy, self.num_pixels_z, self.device
                )
                sum_wepl_gpu += WEPL_img
                sum_count_gpu += count_img

                # Debug visualization (separated inside/outside)
                if hull_rot is not None:
                    WEPL_in, count_in, WEPL_out, count_out = compute_mlp_rigorous(
                        input_data_gpu, self._mlp_params_cache_gpu, hull_rot,
                        self.l_mm, self.image_size_xy_cm, self.d_mm, self.h_mm,
                        self.num_pixels_xy, self.num_pixels_z, self.device,
                        return_debug_images=True
                    )
                    debug_wepl_inside_gpu += WEPL_in
                    debug_count_inside_gpu += count_in
                    debug_wepl_outside_gpu += WEPL_out
                    debug_count_outside_gpu += count_out

                    del WEPL_in, count_in, WEPL_out, count_out

                profiling['total_protons'] += input_data_gpu["positions_u0"].shape[0]
                del WEPL_img, count_img

            print(f" | Process: {time.time()-t_proc:.2f}s")
            print(f"    📊 Protons processed: {profiling['total_protons']:,}")

            del df_full, df_dict, input_data_dict
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Compute averages
        avg_inside_gpu = compute_average_image(debug_wepl_inside_gpu, debug_count_inside_gpu)
        avg_outside_gpu = compute_average_image(debug_wepl_outside_gpu, debug_count_outside_gpu)
        avg_combined_gpu = compute_average_image(sum_wepl_gpu, sum_count_gpu)

        return {
            'inside_gpu': avg_inside_gpu,
            'outside_gpu': avg_outside_gpu,
            'combined_gpu': avg_combined_gpu,
            'hull_rot': hull_rot,
            'sum_wepl': sum_wepl_gpu.cpu().numpy(),
            'sum_count': sum_count_gpu.cpu().numpy(),
            'wepl_inside': debug_wepl_inside_gpu.cpu().numpy(),
            'count_inside': debug_count_inside_gpu.cpu().numpy(),
            'wepl_outside': debug_wepl_outside_gpu.cpu().numpy(),
            'count_outside': debug_count_outside_gpu.cpu().numpy(),
            # GPU tensors for sum (for back-rotation)
            'sum_inside_gpu': debug_wepl_inside_gpu,
            'sum_outside_gpu': debug_wepl_outside_gpu
        }

    def _save_angle_results(
        self,
        angle: int,
        angle_result: Dict,
        hull: Optional[torch.Tensor]
    ) -> None:
        """Save results for one angle"""

        hull_rotated_np = angle_result['hull_rot'].cpu().numpy() if angle_result['hull_rot'] is not None else None

        # Save numpy arrays
        np.save(
            self.output_path / f"proton_paths/average_angle{angle}_degree.npy",
            angle_result['wepl_inside'] + angle_result['wepl_outside']
        )
        np.save(
            self.output_path / f"proton_paths/WEPL_angle{angle}_degree.npy",
            angle_result['wepl_inside'] + angle_result['wepl_outside']
        )
        np.save(
            self.output_path / f"proton_paths/count_angle{angle}_degree.npy",
            angle_result['count_inside'] + angle_result['count_outside']
        )

        # Save debug overlays
        inside_avg = angle_result['inside_gpu'].cpu().numpy()
        outside_avg = angle_result['outside_gpu'].cpu().numpy()

        save_count_debug_overlay(
            angle_result['count_inside'],
            angle_result['count_outside'],
            f"count_angle{angle}_degree",
            self.output_path,
            self.num_pixels_z,
            self.image_size_xy_mm,
            self.num_pixels_xy,
            hull_rotated_np
        )

        save_wepl_debug_overlay(
            angle_result['wepl_inside'],
            angle_result['wepl_outside'],
            f"WEPL_angle{angle}_degree",
            self.output_path,
            self.num_pixels_z,
            self.image_size_xy_mm,
            self.num_pixels_xy,
            hull_rotated_np
        )

        save_average_debug_overlay(
            inside_avg,
            outside_avg,
            f"average_angle{angle}_degree",
            self.output_path,
            self.num_pixels_z,
            self.image_size_xy_mm,
            self.num_pixels_xy,
            hull_rotated_np
        )
