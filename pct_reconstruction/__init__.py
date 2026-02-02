"""
pct_reconstruction - Proton CT Image Reconstruction Package

A modular package for proton CT image reconstruction using hull algorithms.

Main components:
- core: Physics calculations and geometry
- processing: Data loading, trajectory, and projection
- visualization: Plotting and image saving
- utils: Helper functions and utilities
- reconstruction: Main reconstruction class

Example usage:
    from pct_reconstruction import HullImageReconstruction

    reconstructor = HullImageReconstruction(
        csv_paths=['data1.csv', 'data2.csv'],
        output_path='./output',
        device='cuda'
    )

    result = reconstructor.reconstruct(
        use_hull=True,
        hull_geometry='oneFourth_cylinder'
    )
"""

from .reconstruction import HullImageReconstruction

# Optional: expose commonly used functions
from .core import create_hull
from .utils import Timer

__version__ = '1.0.0'

__all__ = [
    'HullImageReconstruction',
    'create_hull',
    'Timer'
]
