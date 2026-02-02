"""
Configuration settings for image reconstruction
"""
from pathlib import Path

# === Paths ===
CSV_FOLDER = "/home/santa/Workspace/image_recon/hull/ProcessedCSV"

# Output folder at repo root (image-reconstruction/output_reconstruction/)
_REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_FOLDER = _REPO_ROOT / "output_reconstruction"

# === Hull Settings ===
USE_HULL = True  # True = ใช้ hull, False = ไม่ใช้
HULL_GEOMETRY = "oneFourth_cylinder"  # หรือ "full_cylinder"

# === Device & Resolution ===
DEVICE = "cuda"  # หรือ "cpu"
NUM_PIXELS_XY = 115
NUM_PIXELS_Z = 115
IMAGE_SIZE_XY_MM = 120.0  # ขนาด ROI ที่ต้องการ reconstruct (mm)
IMAGE_SIZE_Z_MM = 115.0   # ขนาด ROI แกน Z (mm)

# === Physics Parameters ===
EIN_MEV = 100.0      # พลังงานเริ่มต้น (MeV)
INIT_ANGLE = 0       # มุมเริ่มต้น (degrees)
D_MM = 25.0          # ระยะห่าง d (mm)
L_MM = (111.125 - 12.5 + 16) * 2  # ความยาว l (mm)

# === Geometry Parameters ===
RADIUS_CM = 5.0      # รัศมี cylinder (cm)

# === Processing Parameters ===
BATCH_SIZE = 64      # batch size สำหรับ rotate image
