# 🎯 Rigorous MLP Reconstruction - Changes Summary

## สิ่งที่เปลี่ยนแปลง

### ❌ วิธีเก่า (Simplified - ไม่ถูกต้องทางฟิสิกส์)
```python
# ใช้ simplified MLP parameters (G1, G2, H1, H2)
mlp_parameters = get_mlp_parameters(l_mm, num_pixels)

# ใช้ formula ง่ายๆ
WEPL_img, count_img = compute_mlp_img_recon_style(...)

# ปัญหา:
# 1. ใช้ WEPL รวม (ค่าเดียว) กระจายทุก voxel เท่ากัน
# 2. ไม่มีการคำนวณ scattering อย่างเข้มงวด
# 3. ไม่แบ่ง path เป็น 3 ส่วน (straight-MLP-straight)
```

### ✅ วิธีใหม่ (Rigorous - ถูกต้องทางฟิสิกส์)
```python
# ใช้ rigorous scattering matrices (Sigma1, Sigma2, R0, R1)
mlp_parameters = get_mlp_parameters_rigorous(l_mm, num_pixels)

# ใช้ Bayesian MLP estimation
WEPL_img, count_img = compute_mlp_rigorous(...)

# ข้อดี:
# 1. คำนวณ straight trajectories (ก่อน/หลัง hull)
# 2. หาจุดตัด hull (entry/exit) อย่างถูกต้อง
# 3. ใช้ scattering matrices (Sigma) สำหรับ MLP ภายใน hull
# 4. กระจาย WEPL ตาม path length จริง
# 5. Bayesian estimation หา most likely path
```

---

## 📝 ไฟล์ที่แก้ไข

### 1. `core/physics.py`
- ✅ เพิ่ม `get_mlp_parameters_rigorous()`
  - คำนวณ Sigma1, Sigma2, R0, R1 matrices
  - ใช้ Highland formula สำหรับ multiple Coulomb scattering
  - รองรับทั้ง analytical และ numerical integration
  - **CRITICAL FIX**: คำนวณ Sigma(xi, xj) สำหรับ**ทุกคู่ตำแหน่ง** (dynamic lookup)
  - ไม่คำนวณจาก detector (x=0) อีกต่อไป ✅

### 2. `core/__init__.py`
- ✅ Export `get_mlp_parameters_rigorous`

### 3. `processing/trajectory.py`
- ✅ เพิ่ม `compute_mlp_rigorous()` wrapper function
  - Step 1: คำนวณ straight trajectories
  - Step 2: หาจุดตัด hull (entry/exit)
  - Step 3: เรียก `compute_mlp_vectorized_ultra_fast()`
  - Step 4: Scatter results
- ✅ **แก้ไข `compute_mlp_vectorized_ultra_fast()`**:
  - เพิ่ม `start_flat` = hull entry indices
  - เปลี่ยนจาก `Sigma1[x_flat, end_flat]` → `Sigma1[start_flat, x_flat]` ✅
  - เปลี่ยนจาก `Sigma2[x_flat, end_flat]` → `Sigma2[x_flat, end_flat]` (ถูกอยู่แล้ว)
  - **ผลลัพธ์**: Scattering คำนวณเฉพาะ**ภายใน hull** เท่านั้น!

### 4. `processing/__init__.py`
- ✅ Export `compute_mlp_rigorous`

### 5. `reconstruction.py`
- ✅ Import ฟังก์ชันใหม่
- ✅ เปลี่ยนจาก `get_mlp_parameters()` → `get_mlp_parameters_rigorous()`
- ✅ เปลี่ยน MLP cache parameters:
  - เก่า: `X_position, G1, G2, H1, H2`
  - ใหม่: `Sigma1, Sigma2, R0, R1, X1`
- ✅ เปลี่ยนการเรียกใช้:
  - เก่า: `compute_mlp_img_recon_style()`
  - ใหม่: `compute_mlp_rigorous()`

---

## 🔬 ความแตกต่างทางฟิสิกส์

### วิธีเก่า (Simplified):
```
Detector 0 → [MLP Formula] → Detector 2
              (ใช้ G1,G2,H1,H2)
              WEPL เดียวกันทุก voxel ❌
```

### วิธีใหม่ (Rigorous):
```
Detector 0 → [Straight] → Hull Entry → [MLP + Scattering] → Hull Exit → [Straight] → Detector 2
                            ↓                                    ↓
                    find_hull_intersection          Sigma matrices (2×2)
                    (scatter_reduce)                Bayesian estimation
                                                    WEPL distributed by path length ✅
```

---

## 🧮 Scattering Matrices

### Sigma Matrix (Covariance):
```
Σ = [[σ²_y,    σ²_yθ],
     [σ²_yθ,   σ²_θ ]]

where:
σ²_y   = ∫ i_theta(x) × (xf-x)² dx  (position variance)
σ²_θ   = ∫ i_theta(x) dx            (angle variance)
σ²_yθ  = ∫ i_theta(x) × (xf-x) dx   (covariance)
```

### R Matrix (Propagation):
```
R = [[1,  Δx],
     [0,   1]]

where Δx = xf - xi (drift distance)
```

### Highland Formula:
```
i_theta(x) = (E₀/X₀)² × [1 + 0.038 ln(Δx/X₀)]² × f(x)

E₀ = 13.6 MeV
X₀ = 36.1 cm (radiation length of water)
f(x) = polynomial fit
```

---

## 📊 MLP Estimation (Bayesian):

### ❌ วิธีเก่า (ผิด):
```python
# คำนวณ scattering จาก detector (x=0) ← ผิด!
Sigma1 = Sigma(0, x1)      # scattering ในอากาศ ❌
Sigma2 = Sigma(x1, L)      # scattering ในอากาศ ❌
```

### ✅ วิธีใหม่ (ถูก):
```python
# Dynamic lookup ตามจุดเข้า/ออก hull
x_entry = hull_intersection_entry  # จุดเข้า phantom
x_exit = hull_intersection_exit    # จุดออก phantom

Sigma1 = Sigma_cache[x_entry, x1]  # scattering ภายใน phantom เท่านั้น ✅
Sigma2 = Sigma_cache[x1, x_exit]   # scattering ภายใน phantom เท่านั้น ✅
```

### Bayesian Estimation:
```python
# Transform positions
P0_transformed = S_in @ P0
P2_transformed = S_out⁻¹ @ P2

# Bayesian estimation (ใช้ Sigma ที่ถูกต้องแล้ว)
A = Σ₁⁻¹ + R₁ᵀ·Σ₂⁻¹·R₁
B = Σ₁⁻¹·(R₀·P0_transformed) + R₁ᵀ·Σ₂⁻¹·P2_transformed

P_MLP = solve(A, B)  # Most Likely Position
```

---

## 🚀 วิธีใช้งาน

### การรัน (ใช้วิธีใหม่โดยอัตโนมัติ):
```bash
python run_reconstruction.py
```

โค้ดจะใช้ `compute_mlp_rigorous()` โดยอัตโนมัติแล้ว!

### หากต้องการกลับไปใช้วิธีเก่า (ไม่แนะนำ):
แก้ `reconstruction.py`:
```python
# เปลี่ยนกลับเป็น
mlp_parameters = get_mlp_parameters(...)  # แทน get_mlp_parameters_rigorous
WEPL_img = compute_mlp_img_recon_style(...)  # แทน compute_mlp_rigorous
```

---

## 📈 ผลลัพธ์ที่คาดหวัง

### วิธีใหม่จะให้:
- ✅ **ความละเอียดสูงกว่า** - MLP path ถูกต้องกว่า
- ✅ **ขอบเขตชัดกว่า** - hull intersection ถูกต้อง
- ✅ **WEPL distribution ถูกต้อง** - กระจายตาม path length จริง
- ✅ **ฟิสิกส์ถูกต้อง** - scattering matrices + Bayesian estimation

### ข้อแลกเปลี่ยน:
- ⏱️ **ช้ากว่า** - ต้องคำนวณ scattering matrices (แต่ cache ได้)
- 💾 **ใช้ memory มากกว่า** - matrices ขนาด [512×512×2×2]
- 🔧 **ซับซ้อนกว่า** - มีขั้นตอนมากกว่า

---

## 🔍 การตรวจสอบผลลัพธ์

ดูว่า MLP parameters ถูกสร้างหรือยัง:
```bash
ls -lh MLP_parameters_rigorous_*.pkl
```

ดูว่าใช้ scattering matrices หรือไม่:
```bash
python -c "import pickle; d=pickle.load(open('MLP_parameters_rigorous_l_114.625_pixels_512.pkl','rb')); print(d.keys())"
# ควรเห็น: dict_keys(['Sigma1', 'Sigma2', 'R0', 'R1', 'X1'])
```

---

## 📚 References

- **Highland Formula**: Multiple Coulomb Scattering
- **MLP Algorithm**: Most Likely Path estimation
- **Bayesian Estimation**: P_MLP = argmin[(P-P0)ᵀΣ₁⁻¹(P-P0) + (P2-P)ᵀΣ₂⁻¹(P2-P)]

---

## ✅ Status

- [x] สร้าง `get_mlp_parameters_rigorous()`
- [x] สร้าง `compute_mlp_rigorous()`
- [x] แก้ไข `reconstruction.py`
- [x] **แก้ไขปัญหา scattering ในอากาศ** (CRITICAL FIX!)
  - เปลี่ยนจาก `Sigma(0, x)` → `Sigma(x_entry, x)` ✅
  - Scattering คำนวณเฉพาะ**ภายใน hull** เท่านั้น ✅
- [ ] ทดสอบกับข้อมูลจริง
- [ ] เปรียบเทียบผลลัพธ์กับวิธีเก่า

---

## 🎯 สรุปการแก้ไขสำคัญ (CRITICAL FIX)

### ปัญหาเดิม:
Scattering matrices ถูกคำนวณตั้งแต่ **detector (x=0)** ซึ่งเป็น**อากาศ** - ไม่ควรมี scattering!

### การแก้ไข:
1. **`physics.py`**: เปลี่ยนจาก `Sigma(0, x)` → `Sigma(xi, xj)` สำหรับทุกคู่
2. **`trajectory.py`**: ใช้ dynamic lookup: `Sigma[hull_entry, mlp_position]`
3. **ผลลัพธ์**: Scattering คำนวณเฉพาะ**ภายใน phantom** เท่านั้น!

### ตัวอย่าง:
```
❌ เก่า: Detector (x=0) → [scattering in air!] → Hull → [scattering] → Exit
✅ ใหม่: Detector → [no scattering] → Hull Entry → [scattering] → Hull Exit → [no scattering] → Detector
```

---

*Last updated: 2026-01-25 (with CRITICAL FIX for dynamic scattering)*
