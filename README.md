# Module 2 → เบน (Python + OpenCV)

เฮ้ เบน — เราโยก Module 2 จาก MATLAB มา **Python OpenCV** แล้ว  
ของเก่า `.m` ย้ายไปอยู่ `matlab_archive/` ไว้ดูเทียบ ไม่ได้ลบ

เป้าเดิมยังเหมือนเดิม: Live Webcam → ท่า ASL → บอกตัวอักษร A–Z  
เรารับช่วง segmentation / morph / edge ตามเดิม

---

### รันแบบคำสั่งเดียว (แนะนำ)

ดับเบิลคลิก หรือใน cmd จากโฟลเดอร์โปรเจกต์:

```bat
.\setup_and_run.bat
```

สคริปต์จะจัดการให้เอง:
1. สร้าง `.venv` ถ้ายังไม่มี (มีแล้วจะไม่สร้างใหม่)
2. `pip install -r requirements.txt` ใน venv
3. รัน `python -m module2.run_test`

รูปผลไปกองที่ `results_py/` (01–08)  
ของเก่าจาก MATLAB ยังอยู่ที่ `results/` เทียบกันได้

ถ้าเบนอยากรันเองทีละขั้นใน venv:

```bat
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m module2.run_test
```

---

### ฟังก์ชันหลัก 4 ตัว (ใน `module2/`)

| ไฟล์ | ทำอะไร | ส่งอะไรออกมา |
|------|--------|--------------|
| `segment_skin_hsv.py` | แยกมือด้วยสีผิว (HSV) | `binary_mask`, `segmented_img` |
| `segment_otsu.py` | ตัดขาวดำแบบ Otsu | `binary_img`, `level` (ช่วง 0–1) |
| `apply_morphology.py` | ลบจุดรก / อุดรู + **เก็บก้อนใหญ่สุด** | dict มี `cleaned_mask` ฯลฯ |
| `detect_edges_all.py` | หาขอบ Sobel / Prewitt / Canny | dict มี `sobel_edges` ฯลฯ |

อินพุตภาพเป็น **RGB** (HxWx3) เหมือนของ MATLAB เดิม  
ถ้าเบนอ่านด้วย OpenCV (`cv2.imread` เป็น BGR) ต้อง `cvtColor` เป็น RGB ก่อนส่งเข้าฟังก์ชันของเรา

ตัวอย่างเรียกใช้:

```python
from module2 import segment_skin_hsv, apply_morphology, detect_edges_all

limits = {"Hmin": 0, "Hmax": 0.12, "Smin": 0.15, "Smax": 0.75, "Vmin": 0.15, "Vmax": 1}
mask, skin = segment_skin_hsv(rgb, limits)

morph = apply_morphology(mask, se_shape="disk", se_size=5)  # keep_largest=True เป็นค่าเริ่มต้น
edges = detect_edges_all(morph["cleaned_mask"])
```

ถ้าเบนจะผูก GUI:
- HSV ปรับช่วงสีผ่าน dict `hsv_limits` (สเกล 0–1 เหมือนเดิม)
- morph / edge คืนเป็น dict เอาไปโชว์บน canvas ได้เลย
- ถ้าอยาก morph แบบไม่ตัดก้อนเล็กออก: ส่ง `keep_largest=False`

---

### ของทดสอบ / โครงสร้าง

- `setup_and_run.bat` — คำสั่งเดียว: venv + install + เทส
- `.venv/` — environment ของโปรเจกต์ (สร้างอัตโนมัติจาก bat)
- `module2/run_test.py` — รันทั้ง pipeline แล้วเซฟทีละใบ
- `test_hand.jpg` — รูปมือที่เราใช้ลอง
- `results_py/` — ผลรัน Python
- `matlab_archive/` — โค้ด `.m` เก่า (อ้างอิง)
- `results/` — ผลรัน MATLAB เก่า

---

### โน้ตสั้นๆ จากที่เราลอง

HSV จับมือได้ แต่ฉากรก (คีย์บอร์ดฯ) จะมี noise → morph + เก็บก้อนใหญ่สุดช่วยเยอะ  
Otsu ทั้งภาพบนฉากรกไม่ค่อยสวย — เก็บไว้เทียบในรายงาน  
Edge ควรทำหลัง `cleaned_mask` แล้ว จะได้เส้นไม่รก
