"""CSV data loading and preprocessing"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict

from ..core.physics import range_energy_model


def load_csv_fast(path: Path) -> pd.DataFrame:
    """OPTIMIZED: Load CSV with float32 from start.

    Expected columns: CentroidX0, CentroidY0, CentroidX1, CentroidY1,
                     predicted_residue_energy, track_len, offset_mm

    Args:
        path: Path to CSV file

    Returns:
        Cleaned DataFrame with renamed columns
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


def clean_data_batch(
    df_dict: Dict[float, pd.DataFrame],
    device: torch.device
) -> Dict[float, Dict[str, torch.Tensor]]:
    """OPTIMIZED: Batch process all offset groups at once

    Args:
        df_dict: Dictionary mapping offset_mm to DataFrame
        device: PyTorch device (CPU or CUDA)

    Returns:
        Dictionary mapping offset to dict of GPU tensors
        Each inner dict contains: positions_u0-u3, positions_v0-v3, WEPL
    """
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
        wepl_np = range_energy_model(df["E"].values).astype(np.float32)

        # Single GPU transfer per offset
        result[offset] = {
            "positions_u0": torch.from_numpy(u0_np).to(device),
            "positions_u1": torch.from_numpy(u0_np).to(device),
            "positions_u2": torch.from_numpy(u2_np).to(device),
            "positions_u3": torch.from_numpy(u3_np).to(device),
            "positions_v0": torch.from_numpy(v0_np).to(device),
            "positions_v1": torch.from_numpy(v0_np).to(device),
            "positions_v2": torch.from_numpy(v2_np).to(device),
            "positions_v3": torch.from_numpy(v3_np).to(device),
            "WEPL": torch.from_numpy(wepl_np).to(device)
        }

    return result
