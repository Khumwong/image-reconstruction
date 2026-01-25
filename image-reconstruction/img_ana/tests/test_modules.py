#!/usr/bin/env python3
"""
Quick test script to verify all modules import correctly
"""
import sys
from pathlib import Path

# Add parent directory to path (img_ana is in parent of tests/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")

    try:
        # Test main package
        print("  ✓ Importing img_ana...")
        from img_ana import HullImageReconstruction

        # Test core modules
        print("  ✓ Importing core modules...")
        from img_ana.core import (
            get_mlp_parameters_rigorous,
            create_hull,
            rotate_hull
        )

        # Test processing modules
        print("  ✓ Importing processing modules...")
        from img_ana.processing import (
            load_csv_fast,
            clean_data_batch,
            compute_straight_trajectory_batch,
            find_hull_intersection,
            compute_mlp_vectorized_ultra_fast,
            rotate_image_gpu_batched,
            accumulate_projection,
            compute_average_image
        )

        # Test visualization modules
        print("  ✓ Importing visualization modules...")
        from img_ana.visualization import (
            save_wepl_image,
            save_count_image,
            save_average_image,
            save_reimg_image,
            save_reimg_sum_image,
            save_debug_overlay_rotated_hull,
            save_debug_overlay_back_rotated
        )

        # Test utils
        print("  ✓ Importing utils...")
        from img_ana.utils import (
            Timer,
            position_to_index,
            index_to_position,
            extract_angle_from_filename
        )

        print("\n✅ All imports successful!")
        return True

    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without running full reconstruction"""
    print("\nTesting basic functionality...")

    try:
        import torch
        from img_ana.utils import Timer, position_to_index, index_to_position
        from img_ana.core import create_hull

        # Test Timer
        print("  ✓ Testing Timer...")
        with Timer("Test timer", verbose=False) as t:
            import time
            time.sleep(0.1)
        assert t.elapsed > 0.09, "Timer not working"

        # Test coordinate conversion
        print("  ✓ Testing coordinate conversion...")
        pos = 5.0
        idx = position_to_index(pos, 10.0, 1.0)
        pos_back = index_to_position(idx, 10.0, 1.0)
        assert abs(pos - pos_back) < 1e-5, "Coordinate conversion failed"

        # Test hull creation
        print("  ✓ Testing hull creation...")
        device = torch.device('cpu')
        hull = create_hull(
            'oneFourth_cylinder',
            num_pixels_xy=64,
            num_pixels_z=64,
            l_cm=10.0,
            h_cm=10.0,
            device=device,
            radius_cm=3.0
        )
        assert hull.shape == (64, 64, 64), "Hull shape incorrect"
        assert hull.dtype == torch.bool, "Hull dtype incorrect"
        assert hull.any(), "Hull is empty"

        print("\n✅ Basic functionality tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("Module Test Suite for img_ana")
    print("="*70)
    print()

    # Test imports
    import_ok = test_imports()

    # Test basic functionality
    func_ok = test_basic_functionality()

    # Summary
    print("\n" + "="*70)
    if import_ok and func_ok:
        print("✅ ALL TESTS PASSED!")
        print("\nYou can now use the refactored modules:")
        print("  python img_ana/examples/example_using_modules.py --help")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease check the error messages above.")
    print("="*70)


if __name__ == "__main__":
    main()
