#!/usr/bin/env python3
"""
Create MP4 videos from multi-angle reconstruction images

This script creates rotation videos from the output images of the reconstruction process.
It processes all image types (Re_img, Re_img_sum, count, WEPL, average) in both
standard and debug overlay formats.

Usage:
    python create_animations.py [--output-dir OUTPUT_DIR] [--fps FPS] [--show-angle]

Example:
    python create_animations.py --output-dir output_reconstruction --fps 10 --show-angle
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path so we can import pct_reconstruction
sys.path.insert(0, str(Path(__file__).parent.parent))

from pct_reconstruction.core.config import OUTPUT_FOLDER
from typing import List, Tuple, Optional
import re
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import imageio


class AnimationCreator:
    """Creates MP4 animations from multi-angle reconstruction images"""

    def __init__(self, output_dir: str = "output_reconstruction",
                 fps: int = 10, show_angle: bool = True):
        """
        Initialize animation creator

        Args:
            output_dir: Directory containing reconstruction output
            fps: Frames per second for videos
            show_angle: Whether to overlay angle text on frames
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.show_angle = show_angle

        # Animation output folder
        self.animation_dir = self.output_dir / "animation"

        # Image types and their subfolders
        self.image_types = {
            'Re_img': 'Re_img',
            'Re_img_sum': 'Re_img_sum',
            'count': 'count',
            'WEPL': 'WEPL',
            'average': 'average'
        }

        # Validate output directory
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {self.output_dir}")

        # Create animation folder
        self.animation_dir.mkdir(exist_ok=True)

    def extract_angle_from_filename(self, filename: str) -> Optional[int]:
        """Extract angle from filename like 'Re_img_angle3_degree.png'"""
        match = re.search(r'angle(\d+)_degree', filename)
        if match:
            return int(match.group(1))
        return None

    def find_images_by_type(self, image_type: str, subfolder: str) -> List[Tuple[int, Path]]:
        """
        Find all images for a given type and sort by angle

        Args:
            image_type: Type of image (e.g., 'Re_img')
            subfolder: Subfolder containing images

        Returns:
            List of (angle, path) tuples sorted by angle
        """
        folder = self.output_dir / subfolder
        if not folder.exists():
            print(f"  ⚠️  Folder not found: {folder}")
            return []

        images = []

        # Find standard images (without debug_overlay)
        pattern = f"{image_type}_angle*_degree.png"
        for img_path in folder.glob(pattern):
            # Skip debug overlay images
            if 'debug_overlay' in img_path.name:
                continue
            angle = self.extract_angle_from_filename(img_path.name)
            if angle is not None:
                images.append((angle, img_path))

        # Sort by angle
        images.sort(key=lambda x: x[0])
        return images

    def find_debug_images_by_type(self, image_type: str, subfolder: str) -> List[Tuple[int, Path]]:
        """Find all debug overlay images for a given type and sort by angle"""
        folder = self.output_dir / subfolder
        if not folder.exists():
            print(f"  ⚠️  Folder not found: {folder}")
            return []

        images = []

        # Find debug overlay images
        pattern = f"{image_type}_angle*_degree_debug_overlay.png"
        for img_path in folder.glob(pattern):
            angle = self.extract_angle_from_filename(img_path.name)
            if angle is not None:
                images.append((angle, img_path))

        # Sort by angle
        images.sort(key=lambda x: x[0])
        return images

    def add_angle_text(self, img: Image.Image, angle: int) -> Image.Image:
        """
        Add angle text overlay to frame

        Args:
            img: PIL Image
            angle: Angle in degrees

        Returns:
            Image with text overlay
        """
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        text = f"Angle: {angle}°"

        # Try to use a nice font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
            except:
                font = ImageFont.load_default()

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position: top-right corner with padding
        padding = 20
        x = img.width - text_width - padding
        y = padding

        # Draw background rectangle
        draw.rectangle(
            [x - 10, y - 10, x + text_width + 10, y + text_height + 10],
            fill=(0, 0, 0)
        )

        # Draw text
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

        return img_copy

    def create_video(self, images: List[Tuple[int, Path]],
                    output_filename: str, video_type: str) -> bool:
        """
        Create MP4 video from list of images

        Args:
            images: List of (angle, path) tuples
            output_filename: Output video filename
            video_type: Type description for logging

        Returns:
            True if successful, False otherwise
        """
        if not images:
            print(f"  ⚠️  No images found for {video_type}")
            return False

        print(f"  Creating {video_type}... ({len(images)} frames)")

        output_path = self.animation_dir / output_filename

        try:
            # First pass: load all images and find max dimensions
            all_images = []
            max_width = 0
            max_height = 0

            for angle, img_path in images:
                try:
                    # Read image with PIL
                    img = Image.open(str(img_path))

                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    all_images.append((angle, img))
                    max_width = max(max_width, img.width)
                    max_height = max(max_height, img.height)

                except Exception as e:
                    print(f"  ⚠️  Skipping unreadable image: {img_path} ({e})")
                    continue

            if not all_images:
                print(f"  ❌ No valid frames found")
                return False

            # Second pass: resize all to max dimensions and add text
            frames = []
            for angle, img in all_images:
                # Resize to max dimensions (pad with black if needed)
                if img.width != max_width or img.height != max_height:
                    new_img = Image.new('RGB', (max_width, max_height), (0, 0, 0))
                    # Center the image
                    x_offset = (max_width - img.width) // 2
                    y_offset = (max_height - img.height) // 2
                    new_img.paste(img, (x_offset, y_offset))
                    img = new_img

                # Add angle text if requested
                if self.show_angle:
                    img = self.add_angle_text(img, angle)

                # Convert to numpy array
                frames.append(np.array(img))

            # Write video using imageio with ffmpeg
            with imageio.get_writer(
                str(output_path),
                fps=self.fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p',
                format='FFMPEG'
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)

            # Get file size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ Saved: {output_filename} ({file_size_mb:.2f} MB, {len(frames)} frames)")

            return True

        except Exception as e:
            print(f"  ❌ Failed to create video: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_all_videos(self):
        """Create all videos for all image types"""
        print("="*70)
        print("Creating Animation Videos")
        print("="*70)
        print(f"Source directory: {self.output_dir}")
        print(f"Animation output: {self.animation_dir}")
        print(f"FPS: {self.fps}")
        print(f"Show angle overlay: {self.show_angle}")
        print()

        success_count = 0
        total_count = 0

        for image_type, subfolder in self.image_types.items():
            print(f"Processing {image_type}...")

            # Standard images
            images = self.find_images_by_type(image_type, subfolder)
            if images:  # Only create video if images exist
                total_count += 1
                output_filename = f"{image_type}_rotation.mp4"
                if self.create_video(images, output_filename, f"{image_type} (standard)"):
                    success_count += 1
            else:
                print(f"  ⚠️  No standard images found, skipping {image_type}_rotation.mp4")

            # Debug overlay images
            debug_images = self.find_debug_images_by_type(image_type, subfolder)
            if debug_images:  # Only create video if images exist
                total_count += 1
                debug_output_filename = f"{image_type}_debug_rotation.mp4"
                if self.create_video(debug_images, debug_output_filename, f"{image_type} (debug)"):
                    success_count += 1
            else:
                print(f"  ⚠️  No debug overlay images found, skipping {image_type}_debug_rotation.mp4")

            print()

        print("="*70)
        print(f"Summary: {success_count}/{total_count} videos created successfully")
        print("="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Create MP4 rotation videos from reconstruction images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings
  python create_animations.py

  # Custom output directory and FPS
  python create_animations.py --output-dir my_output --fps 15

  # Without angle overlay
  python create_animations.py --no-angle
        """
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(OUTPUT_FOLDER),
        help=f'Directory containing reconstruction output (default: {OUTPUT_FOLDER})'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second for videos (default: 10)'
    )

    parser.add_argument(
        '--show-angle',
        action='store_true',
        default=True,
        help='Show angle text overlay on frames (default: True)'
    )

    parser.add_argument(
        '--no-angle',
        action='store_false',
        dest='show_angle',
        help='Disable angle text overlay'
    )

    args = parser.parse_args()

    try:
        creator = AnimationCreator(
            output_dir=args.output_dir,
            fps=args.fps,
            show_angle=args.show_angle
        )
        creator.create_all_videos()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
