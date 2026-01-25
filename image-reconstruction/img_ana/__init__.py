"""
img_ana - Modular Image Reconstruction Package

A refactored modular version of hull_fast_optimized.py for easier maintenance
and modification.

Main components:
- core: Physics calculations and geometry
- processing: Data loading, trajectory, and projection
- visualization: Plotting and image saving
- utils: Helper functions and utilities
- reconstruction: Main reconstruction class

Example usage:
    from img_ana import HullImageReconstruction

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
from .core import get_mlp_parameters, create_hull
from .utils import Timer

__version__ = '1.0.0'

__all__ = [
    'HullImageReconstruction',
    'get_mlp_parameters',
    'create_hull',
    'Timer'
]
