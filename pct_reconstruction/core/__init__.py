"""Core physics and geometry modules"""
from .physics import (
    range_energy_model,
    get_mlp_parameters_rigorous
)
from .geometry import (
    create_hull,
    rotate_hull
)
from .config import (
    CSV_FOLDER, OUTPUT_FOLDER,
    USE_HULL, HULL_GEOMETRY,
    DEVICE, NUM_PIXELS_XY, NUM_PIXELS_Z,
    IMAGE_SIZE_XY_MM, IMAGE_SIZE_Z_MM,
    EIN_MEV, INIT_ANGLE, D_MM, L_MM,
    RADIUS_CM, BATCH_SIZE
)

__all__ = [
    'range_energy_model',
    'get_mlp_parameters_rigorous',
    'create_hull',
    'rotate_hull',
    # Config
    'CSV_FOLDER', 'OUTPUT_FOLDER',
    'USE_HULL', 'HULL_GEOMETRY',
    'DEVICE', 'NUM_PIXELS_XY', 'NUM_PIXELS_Z',
    'IMAGE_SIZE_XY_MM', 'IMAGE_SIZE_Z_MM',
    'EIN_MEV', 'INIT_ANGLE', 'D_MM', 'L_MM',
    'RADIUS_CM', 'BATCH_SIZE'
]
