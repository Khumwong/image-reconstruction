# pct_reconstruction - Modular Image Reconstruction Package



## 📑 Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Get started quickly
- **[Refactoring Summary](docs/REFACTORING_SUMMARY.md)** - Detailed refactoring documentation
- **[Structure Overview](docs/pct_reconstruction_structure.txt)** - Module structure diagram

## 📁 Structure

```
pct_reconstruction/
├── __init__.py              # Main package interface
├── reconstruction.py        # Main reconstruction orchestrator class
├── core/                    # Physics and geometry
│   ├── __init__.py
│   ├── physics.py          # MLP calculations, scattering matrices
│   └── geometry.py         # Hull creation and transformations
├── processing/             # Data processing
│   ├── __init__.py
│   ├── data_loader.py     # CSV loading and cleaning
│   ├── trajectory.py      # Trajectory and MLP path calculations
│   └── projection.py      # Back-projection operations
├── visualization/          # Plotting
│   ├── __init__.py
│   └── plotting.py        # Image saving and debug overlays
└── utils/                  # Utilities
    ├── __init__.py
    ├── timer.py           # Timer class
    └── helpers.py         # Helper functions
```

## 🚀 Quick Start

### Basic Usage

```python
from pct_reconstruction import HullImageReconstruction

# Create reconstructor
reconstructor = HullImageReconstruction(
    csv_paths=['data1.csv', 'data2.csv'],
    output_path='./output',
    device='cuda',
    num_pixels_xy=512,
    num_pixels_z=512
)

# Run reconstruction
result = reconstructor.reconstruct(
    use_hull=True,
    hull_geometry='oneFourth_cylinder'
)

# Save result
import numpy as np
np.save('output/reconstruction.npy', result)
```

### Command Line

```bash
# Using the example script
python pct_reconstruction/examples/example_using_modules.py --csv_folder ./data --output ./output

# With custom settings
python pct_reconstruction/examples/example_using_modules.py \
    --csv_folder ./data \
    --output ./output \
    --device cuda \
    --resolution_xy 512 \
    --resolution_z 512 \
    --hull_geometry oneFourth_cylinder
```

## 🔧 Module Details

### 1. Core Modules (`core/`)

#### `physics.py`
- `get_mlp_parameters_rigorous()` - Compute/load MLP scattering parameters
- `range_energy_model()` - Convert energy to WEPL
- `compute_sigma_matrix()` - Scattering covariance matrices
- `analytical_sigma_sq_n()` - Fast analytical integration (1000x speedup)

#### `geometry.py`
- `create_hull()` - Generate 3D hull geometry
- `rotate_hull()` - Rotate hull for different angles

### 2. Processing Modules (`processing/`)

#### `data_loader.py`
- `load_csv_fast()` - Optimized CSV loading
- `clean_data_batch()` - Batch data preprocessing

#### `trajectory.py`
- `compute_straight_trajectory_batch()` - GPU-accelerated trajectory calculation
- `find_hull_intersection()` - Find entry/exit points
- `compute_mlp_vectorized_ultra_fast()` - **Main MLP calculation (NO LOOPS!)**

#### `projection.py`
- `rotate_image_gpu_batched()` - GPU-based image rotation
- `accumulate_projection()` - Scatter-add operations
- `compute_average_image()` - Average WEPL calculation

### 3. Visualization (`visualization/`)

#### `plotting.py`
- `save_image_fast()` - Quick image export
- `save_debug_overlay_rotated_hull()` - Debug visualization with hull
- `save_debug_overlay_back_rotated()` - Back-rotated debug images

### 4. Utils (`utils/`)

#### `timer.py`
- `Timer` - Context manager for profiling

#### `helpers.py`
- `position_to_index()` - Coordinate conversion
- `index_to_position()` - Inverse coordinate conversion
- `extract_angle_from_filename()` - Parse angle from CSV filename

## 🎯 Key Features

✅ **Modular Design** - Easy to modify individual components
✅ **Type Hints** - Full type annotations for better IDE support
✅ **Documentation** - Comprehensive docstrings
✅ **GPU Optimized** - Pure GPU vectorization, no Python loops
✅ **Mixed Precision** - FP16 support for Tensor Cores
✅ **Maintainable** - Each module < 400 lines

## 📝 Example: Modifying Physics

Want to change the scattering model? Just edit `core/physics.py`:

```python
# core/physics.py

def analytical_sigma_sq_n(xi: float, xf: float, n: int) -> float:
    """Your custom scattering calculation here"""
    # Modify this function
    pass
```

## 📝 Example: Custom Visualization

Need different plot style? Edit `visualization/plotting.py`:

```python
# visualization/plotting.py

def save_image_fast(array, filename, output_path, ...):
    """Change colormap, labels, etc."""
    plt.imshow(array, cmap='viridis')  # Change here
```

## 🆚 Comparison with Original

| Feature | Original (`hull_fast_optimized.py`) | Refactored (`pct_reconstruction/`) |
|---------|-------------------------------------|-------------------------|
| Lines of code | 1446 lines (single file) | ~200 lines per module |
| Maintainability | Hard to navigate | Easy to find/modify |
| Reusability | Monolithic class | Composable functions |
| Testing | Difficult | Each module testable |
| Documentation | Inline comments | Module-level docs |

## 🔄 Migration Guide

### Old Way
```python
from hull_fast_optimized import HullImageReconstructionFastOptimized

ir = HullImageReconstructionFastOptimized(...)
result = ir.fBackProjection(...)
```

### New Way
```python
from pct_reconstruction import HullImageReconstruction

ir = HullImageReconstruction(...)
result = ir.reconstruct(...)
```

## 📦 Dependencies

- numpy
- pandas
- torch (PyTorch)
- scipy
- matplotlib
- scikit-image (for contour detection)

## 🐛 Troubleshooting

### Import Error
```python
# Make sure you're in the correct directory
import sys
sys.path.insert(0, '/path/to/hull')
from pct_reconstruction import HullImageReconstruction
```

### CUDA Out of Memory
```python
# Reduce resolution or use CPU
reconstructor = HullImageReconstruction(
    ...,
    num_pixels_xy=256,  # Reduce from 512
    device='cpu'
)
```

## 📄 License

Same as original codebase.

## 👥 Authors

Refactored from `hull_fast_optimized.py` into modular architecture.
