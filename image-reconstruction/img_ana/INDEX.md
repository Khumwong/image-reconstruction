# 📑 img_ana Package Index

Complete self-contained modular image reconstruction package.

## 📁 Directory Structure

```
img_ana/
├── 📂 core/                      Physics & Geometry
│   ├── physics.py               MLP parameters, scattering
│   └── geometry.py              Hull creation & rotation
│
├── 📂 processing/                Data Pipeline
│   ├── data_loader.py           CSV loading & cleaning
│   ├── trajectory.py            Trajectory & MLP calculations
│   └── projection.py            Back-projection operations
│
├── 📂 visualization/             Plotting & Visualization
│   └── plotting.py              Image saving & debug overlays
│
├── 📂 utils/                     Utilities
│   ├── timer.py                 Performance profiling
│   └── helpers.py               Coordinate transforms, parsing
│
├── 📂 examples/                  Usage Examples
│   └── example_using_modules.py Working example script
│
├── 📂 tests/                     Test Suite
│   └── test_modules.py          Module import & functionality tests
│
├── 📂 docs/                      Documentation
│   ├── QUICK_START.md           Quick start guide
│   ├── REFACTORING_SUMMARY.md   Detailed refactoring docs
│   └── img_ana_structure.txt    ASCII structure diagram
│
├── reconstruction.py             Main reconstruction orchestrator
├── __init__.py                   Package interface
├── README.md                     Main documentation
└── INDEX.md                      This file

```

## 🚀 Getting Started

### 1. Test the Installation
```bash
cd /home/sutpct/Workspace/img_recon/hull
python img_ana/tests/test_modules.py
```

### 2. Run Example
```bash
python img_ana/examples/example_using_modules.py --help
```

### 3. Use in Your Code
```python
from img_ana import HullImageReconstruction

reconstructor = HullImageReconstruction(
    csv_paths=['data.csv'],
    output_path='./output',
    device='cuda'
)
result = reconstructor.reconstruct(use_hull=True)
```

## 📚 Documentation Files

| File | Description |
|------|-------------|
| [README.md](README.md) | Main package documentation |
| [docs/QUICK_START.md](docs/QUICK_START.md) | Quick start guide (Thai) |
| [docs/REFACTORING_SUMMARY.md](docs/REFACTORING_SUMMARY.md) | Detailed refactoring documentation |
| [docs/img_ana_structure.txt](docs/img_ana_structure.txt) | ASCII structure diagram |
| [INDEX.md](INDEX.md) | This index file |

## 🔧 Module Breakdown

### Core Modules (309 lines)
- **[core/physics.py](core/physics.py)** (220 lines)
  - MLP parameter computation
  - Scattering physics calculations
  - Fast analytical integration

- **[core/geometry.py](core/geometry.py)** (89 lines)
  - Hull geometry creation
  - 3D rotations

### Processing Modules (716 lines)
- **[processing/data_loader.py](processing/data_loader.py)** (95 lines)
  - CSV file loading
  - Data cleaning & WEPL conversion

- **[processing/trajectory.py](processing/trajectory.py)** (520 lines)
  - Straight trajectory calculation
  - Hull intersection detection
  - MLP path computation (vectorized)

- **[processing/projection.py](processing/projection.py)** (101 lines)
  - GPU image rotation
  - Projection accumulation
  - Average image computation

### Visualization (310 lines)
- **[visualization/plotting.py](visualization/plotting.py)** (310 lines)
  - Fast image saving
  - Debug overlays
  - Hull boundary drawing

### Utilities (77 lines)
- **[utils/timer.py](utils/timer.py)** (24 lines)
  - Performance profiling
  - Context manager for timing

- **[utils/helpers.py](utils/helpers.py)** (53 lines)
  - Coordinate transformations
  - Filename parsing

### Main Orchestrator (520 lines)
- **[reconstruction.py](reconstruction.py)** (520 lines)
  - Main reconstruction pipeline
  - Combines all modules
  - High-level API

## 🧪 Testing

Run the test suite:
```bash
python img_ana/tests/test_modules.py
```

Expected output:
```
✅ ALL TESTS PASSED!
```

## 📖 Usage Examples

### Basic Usage
```python
from img_ana import HullImageReconstruction

reconstructor = HullImageReconstruction(
    csv_paths=['data1.csv', 'data2.csv'],
    output_path='./output',
    device='cuda',
    num_pixels_xy=512,
    num_pixels_z=512
)

result = reconstructor.reconstruct(
    use_hull=True,
    hull_geometry='oneFourth_cylinder'
)
```

### Using Individual Modules
```python
# Use only physics module
from img_ana.core import get_mlp_parameters_rigorous

params = get_mlp_parameters_rigorous(
    l_cm=11.5,
    num_pixels=512,
    use_fast=True
)

# Use only geometry module
from img_ana.core import create_hull

hull = create_hull(
    geometry='oneFourth_cylinder',
    num_pixels_xy=512,
    num_pixels_z=512,
    l_cm=11.5,
    h_cm=11.5,
    device=device
)
```

## 🔄 Package Structure Benefits

✅ **Self-contained** - All code, docs, tests, examples in one folder
✅ **Modular** - Each component is independent
✅ **Documented** - Comprehensive docs included
✅ **Tested** - Automated test suite included
✅ **Examples** - Working examples included
✅ **Maintainable** - Easy to find and modify code

## 📊 Statistics

| Metric | Value |
|--------|-------|
| Total modules | 11 |
| Total lines (code) | ~1,932 |
| Original file size | 1,446 lines |
| Largest module | 520 lines (trajectory.py) |
| Average module size | ~150 lines |
| Documentation files | 4 |
| Example scripts | 1 |
| Test files | 1 |

## 🎯 Common Tasks

### Modify Physics Model
Edit: [core/physics.py](core/physics.py)

### Change Visualization Style
Edit: [visualization/plotting.py](visualization/plotting.py)

### Add New Hull Shape
Edit: [core/geometry.py](core/geometry.py)

### Customize Data Loading
Edit: [processing/data_loader.py](processing/data_loader.py)

### Adjust Reconstruction Pipeline
Edit: [reconstruction.py](reconstruction.py)

## 📞 Support

All resources are included in this package:

- **Documentation**: See `docs/` folder
- **Examples**: See `examples/` folder
- **Tests**: See `tests/` folder
- **Main README**: [README.md](README.md)

---

**Package Status**: ✅ Complete and Ready to Use

**Original**: `hull_fast_optimized.py` (1446 lines)
**Refactored**: 11 focused modules in self-contained package
