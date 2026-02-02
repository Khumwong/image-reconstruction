"""Helper functions for coordinate transformations and file operations"""
import re
from pathlib import Path
from typing import Optional


def position_to_index(position: float, l: float, pixel_size: float) -> float:
    """Convert physical position to pixel index

    Args:
        position: Physical position in cm
        l: Total length in cm
        pixel_size: Size of one pixel in cm

    Returns:
        Pixel index (float)
    """
    return (position + l/2) / pixel_size


def index_to_position(index: float, l: float, pixel_size: float) -> float:
    """Convert pixel index to physical position

    Args:
        index: Pixel index
        l: Total length in cm
        pixel_size: Size of one pixel in cm

    Returns:
        Physical position in cm
    """
    return (index * pixel_size) - (l/2)


def extract_angle_from_filename(path: Path) -> Optional[int]:
    """Extract angle directly from filename.

    Pattern: results_run506113706_251213113712_degree_90.csv

    Args:
        path: Path to CSV file

    Returns:
        angle (int) or None if not found
    """
    pattern = re.compile(r"results_run\d+_\d+_degree_(\d+)\.csv$", re.IGNORECASE)
    match = pattern.search(path.name)
    if match:
        return int(match.group(1))
    return None
