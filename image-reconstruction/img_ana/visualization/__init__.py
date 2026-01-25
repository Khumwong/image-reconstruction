"""Visualization and plotting functions"""
from .plotting import (
    get_white_viridis_cmap,
    # Simple image functions
    save_reimg_image,
    save_reimg_sum_image,
    # Debug overlay functions (new)
    save_average_debug_overlay,
    save_count_debug_overlay,
    save_wepl_debug_overlay,
    save_reimg_debug_overlay,
    save_reimg_sum_debug_overlay,
    # Deprecated functions (kept for backward compatibility)
    save_wepl_image,
    save_count_image,
    save_average_image,
)

__all__ = [
    'get_white_viridis_cmap',
    # Simple image functions
    'save_reimg_image',
    'save_reimg_sum_image',
    # Debug overlay functions (new)
    'save_average_debug_overlay',
    'save_count_debug_overlay',
    'save_wepl_debug_overlay',
    'save_reimg_debug_overlay',
    'save_reimg_sum_debug_overlay',
    # Deprecated functions (kept for backward compatibility)
    'save_wepl_image',
    'save_count_image',
    'save_average_image',
]
