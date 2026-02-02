# 📖 คู่มือการใช้งาน pct_reconstruction - ฉบับสมบูรณ์

## 🎯 Scripts ที่มีให้ใช้

มี 3 scripts สำคัญ:

### 1. **`run_reconstruction.py`** - รัน reconstruction ใหม่
### 2. **`check_output.py`** - ตรวจสอบ output ที่มีอยู่
### 3. **`pct_reconstruction/tests/test_modules.py`** - ทดสอบว่า modules ใช้งานได้

---

## 🚀 วิธีใช้งาน

### ขั้นตอนที่ 1: ตรวจสอบว่ามี Output อยู่แล้วหรือไม่

```bash
cd /home/sutpct/Workspace/img_recon/hull
python check_output.py
```

**ผลลัพธ์ที่คุณจะเห็น:**
```
✓ Found 5 output folder(s)

📂 Folder 1: ./out_img_recon
   📁 proton_paths/: 183 .npy files
   🖼️  proton_paths_images_2/: 367 .png images
   ⭐ BackProjection_hull.npy
      Shape: (512, 512, 512)
   💾 Total size: 92.0 GB

... (และอีก 4 folders)
```

**คุณมี output folders อยู่แล้ว 5 folders!**

---

### ขั้นตอนที่ 2: ใช้ Output ที่มีอยู่

#### **2.1 โหลดผลลัพธ์ใน Python**

```python
import numpy as np

# โหลดผลลัพธ์
result = np.load('./out_img_recon/BackProjection_hull.npy')

print(f"Shape: {result.shape}")        # (512, 512, 512)
print(f"Min: {result.min():.2f}")      # 0.00
print(f"Max: {result.max():.2f}")      # 2508.82
print(f"Mean: {result.mean():.2f}")
```

#### **2.2 ดูภาพ PNG ที่สร้างไว้**

```bash
# ดูภาพทั้งหมด
ls ./out_img_recon/proton_paths_images_2/

# เปิดภาพด้วย image viewer
eog ./out_img_recon/proton_paths_images_2/hull.png
eog ./out_img_recon/proton_paths_images_2/Re_img_angle0_degree.png
```

#### **2.3 โหลดข้อมูล WEPL, count, average**

```python
import numpy as np

# โหลดข้อมูลแต่ละมุม
angle = 0
wepl = np.load(f'./out_img_recon/proton_paths/WEPL_angle{angle}_degree.npy')
count = np.load(f'./out_img_recon/proton_paths/count_angle{angle}_degree.npy')
average = np.load(f'./out_img_recon/proton_paths/average_angle{angle}_degree.npy')

print(f"WEPL shape: {wepl.shape}")
print(f"Count shape: {count.shape}")
print(f"Average shape: {average.shape}")
```

---

### ขั้นตอนที่ 3: รัน Reconstruction ใหม่ (ถ้าต้องการ)

#### **3.1 ใช้ Script พร้อมใช้งาน**

```bash
cd /home/sutpct/Workspace/img_recon/hull
python run_reconstruction.py
```

Script จะ:
1. ✅ หา CSV files อัตโนมัติ
2. ✅ สร้าง output folder ใหม่
3. ✅ รัน reconstruction
4. ✅ บันทึกผลลัพธ์

**แก้ไข Configuration:**

เปิดไฟล์ `run_reconstruction.py` แก้บรรทัดเหล่านี้:

```python
# ========== CONFIGURATION ==========
CSV_FOLDER = "/path/to/your/csv/files"  # เปลี่ยนตรงนี้
OUTPUT_FOLDER = "./my_output"            # เปลี่ยนชื่อ output
USE_HULL = True                          # True/False
HULL_GEOMETRY = "oneFourth_cylinder"     # หรือ "full_cylinder"
DEVICE = "cuda"                          # หรือ "cpu"
RESOLUTION_XY = 512                      # ลดถ้า GPU ไม่พอ
RESOLUTION_Z = 512
# ===================================
```

#### **3.2 หรือเขียนโค้ดเอง**

สร้างไฟล์ใหม่ `my_reconstruction.py`:

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/sutpct/Workspace/img_recon/hull')

from pct_reconstruction import HullImageReconstruction
import glob
import numpy as np

# หา CSV files
csv_files = glob.glob('/path/to/csv/*.csv')
print(f"Found {len(csv_files)} files")

# สร้าง reconstructor
reconstructor = HullImageReconstruction(
    csv_paths=csv_files,
    output_path='./my_output',
    device='cuda',
    num_pixels_xy=512,
    num_pixels_z=512
)

# รัน
result = reconstructor.reconstruct(use_hull=True)

# บันทึก
np.save('./my_output/result.npy', result)
print("✅ Done!")
```

รัน:
```bash
python my_reconstruction.py
```

---

## 📁 โครงสร้าง Output Folder

เมื่อรัน reconstruction เสร็จ จะได้:

```
output_folder/
├── proton_paths/                      # ข้อมูล numpy
│   ├── WEPL_angle0_degree.npy        # WEPL สำหรับแต่ละมุม
│   ├── count_angle0_degree.npy       # จำนวนโปรตอนที่ผ่าน
│   ├── average_angle0_degree.npy     # ค่าเฉลี่ย
│   ├── WEPL_angle45_degree.npy
│   └── ... (สำหรับทุกมุม)
│
├── proton_paths_2/                    # (โครงสร้างเดียวกัน)
│
├── proton_paths_images_2/             # ภาพ PNG
│   ├── hull.png                      # ภาพ hull
│   ├── Re_img_angle0_degree.png      # ภาพ reconstruction
│   ├── count_angle0_degree_debug_overlay.png  # Debug overlay
│   └── ... (หลายภาพ)
│
└── BackProjection_hull.npy            # ⭐ ผลลัพธ์สุดท้าย (512×512×512)
```

---

## 💡 Use Cases ต่างๆ

### Use Case 1: ดูผลลัพธ์ที่มีอยู่แล้ว

```python
import numpy as np
import matplotlib.pyplot as plt

# โหลด
result = np.load('./out_img_recon/BackProjection_hull.npy')

# Plot ภาพตัด
slice_idx = 256  # ตรงกลาง
plt.figure(figsize=(10, 10))
plt.imshow(result[:, :, slice_idx], cmap='gray')
plt.title(f'Slice {slice_idx}')
plt.colorbar()
plt.show()
```

### Use Case 2: เปรียบเทียบ Output หลาย Folders

```python
import numpy as np

# โหลดจากหลาย folders
result1 = np.load('./out_img_recon/BackProjection_hull.npy')
result2 = np.load('./out_img_recon2/BackProjection_hull.npy')

# เปรียบเทียบ
diff = np.abs(result1 - result2)
print(f"Max difference: {diff.max():.2f}")
print(f"Mean difference: {diff.mean():.2f}")
```

### Use Case 3: Extract Specific Slice

```python
import numpy as np

result = np.load('./out_img_recon/BackProjection_hull.npy')

# Extract slice ที่ต้องการ
xy_slice = result[:, :, 256]    # XY plane
xz_slice = result[:, 256, :]    # XZ plane
yz_slice = result[256, :, :]    # YZ plane

# บันทึก
np.save('slice_xy.npy', xy_slice)
np.save('slice_xz.npy', xz_slice)
np.save('slice_yz.npy', yz_slice)
```

### Use Case 4: รัน Reconstruction ด้วยค่าต่างๆ

```python
import sys
sys.path.insert(0, '/home/sutpct/Workspace/img_recon/hull')

from pct_reconstruction import HullImageReconstruction

csv_files = ['data1.csv', 'data2.csv']

# ทดลองหลายแบบ
configs = [
    {'use_hull': True, 'geometry': 'oneFourth_cylinder'},
    {'use_hull': True, 'geometry': 'full_cylinder'},
    {'use_hull': False, 'geometry': None},
]

for i, config in enumerate(configs):
    print(f"Running config {i+1}...")

    reconstructor = HullImageReconstruction(
        csv_paths=csv_files,
        output_path=f'./output_test_{i+1}',
        device='cuda'
    )

    result = reconstructor.reconstruct(
        use_hull=config['use_hull'],
        hull_geometry=config['geometry']
    )

    print(f"✅ Config {i+1} done")
```

---

## 🛠️ Troubleshooting

### ปัญหา 1: ไม่พบ CSV files

```bash
# ตรวจสอบว่า CSV folder มีไฟล์
ls /home/sutpct/Workspace/test_pyeudaq_reader/ProcessedCSV/*.csv
```

### ปัญหา 2: GPU memory ไม่พอ

แก้ใน `run_reconstruction.py`:
```python
RESOLUTION_XY = 256  # ลดจาก 512
RESOLUTION_Z = 256
# หรือ
DEVICE = "cpu"  # ใช้ CPU แทน
```

### ปัญหา 3: Import error

```python
# ตรวจสอบว่า path ถูกต้อง
import sys
sys.path.insert(0, '/home/sutpct/Workspace/img_recon/hull')

# ทดสอบ import
from pct_reconstruction import HullImageReconstruction
print("✅ Import successful!")
```

---

## 📊 ข้อมูลเพิ่มเติม

### Output Folders ที่คุณมีอยู่

จากการ scan พบว่ามี **5 folders**:

1. `./out_img_recon` - 92.0 GB (183 .npy, 367 .png)
2. `./out_img_recon2` - 92.1 GB (183 .npy, 305 .png)
3. `./out_img_recon3` - 92.1 GB (183 .npy, 305 .png)
4. `./output_hull_fast_optimized` - 92.1 GB (183 .npy, 306 .png)
5. `./output_hull_optimized` - 3.0 GB (6 .npy, 9 .png)

**แนะนำ:** ใช้ folder ที่มีข้อมูลครบที่สุด (folders 1-4)

---

## 📚 เอกสารเพิ่มเติม

- [pct_reconstruction/README.md](pct_reconstruction/README.md) - API documentation
- [pct_reconstruction/docs/QUICK_START.md](pct_reconstruction/docs/QUICK_START.md) - Quick guide
- [pct_reconstruction/COMPLETE.md](pct_reconstruction/COMPLETE.md) - Package status

---

## 🎯 สรุป

### ถ้าต้องการ**ใช้ผลลัพธ์ที่มีอยู่**:
```bash
python check_output.py  # ดูว่ามีอะไรบ้าง
```
```python
import numpy as np
result = np.load('./out_img_recon/BackProjection_hull.npy')
```

### ถ้าต้องการ**รัน reconstruction ใหม่**:
```bash
python run_reconstruction.py  # รันทันที
```

### ถ้าต้องการ**แก้ไขโค้ด**:
```python
from pct_reconstruction import HullImageReconstruction
# เขียนโค้ดเอง
```

---

**คุณมีทั้ง output ที่พร้อมใช้และ tools ครบสำหรับรัน reconstruction ใหม่!** 🎉
