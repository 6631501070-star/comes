"""Module 3 — Gesture Classifier (ASL Recognition using OpenCV + SVM)

หน้าที่:
  1. สกัด Feature จากภาพมือ (Normalized Silhouette + Custom HOG + Spatial Density + Hu Moments)
  2. เทรนโมเดล Support Vector Machine (SVM) จากโฟลเดอร์ dataset/
  3. บันทึกและโหลดโมเดล (.pkl) สำหรับนำไปใช้ใน Real-time Webcam / GUI
"""

from __future__ import annotations
from pathlib import Path
import pickle
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from module2.segment_skin_hsv import segment_skin_hsv
from module2.apply_morphology import apply_morphology

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = _ROOT / "gesture_model.pkl"


def extract_hand_mask(rgb_img: np.ndarray) -> np.ndarray:
    """ใช้ Pipeline ของ Module 2 สกัด Cleaned Mask ของมือ"""
    limits = {
        "Hmin": 0.0,
        "Hmax": 0.12,
        "Smin": 0.15,
        "Smax": 0.75,
        "Vmin": 0.15,
        "Vmax": 1.0,
    }
    mask, _ = segment_skin_hsv(rgb_img, limits)
    morph = apply_morphology(mask, se_shape="disk", se_size=4, keep_largest=True)
    return morph["cleaned_mask"].astype(np.uint8) * 255


def _compute_custom_hog(norm_img: np.ndarray) -> np.ndarray:
    """คำนวณ HOG บนภาพ 64x64 โดยแบ่งเป็นเซลล์ 16x16 (4x4 = 16 เซลล์) เซลล์ละ 8 bins = 128 มิติ"""
    gx = cv2.Sobel(norm_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm_img, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    hog_vec = []
    cell_sz = 16
    for i in range(0, 64, cell_sz):
        for j in range(0, 64, cell_sz):
            c_mag = mag[i : i + cell_sz, j : j + cell_sz]
            c_ang = ang[i : i + cell_sz, j : j + cell_sz]
            hist, _ = np.histogram(c_ang, bins=8, range=(0, 360), weights=c_mag)
            hog_vec.extend(hist)

    hog_arr = np.array(hog_vec, dtype=np.float32)
    norm = np.linalg.norm(hog_arr) + 1e-6
    return hog_arr / norm


def extract_features_from_mask(mask_u8: np.ndarray) -> np.ndarray | None:
    """
    Data Normalization & Feature Extraction:
    1. Bounding Box Crop เอาเฉพาะส่วนมือ
    2. Resize เป็น 64x64 (Scale Invariance)
    3. คำนวณ HOG (128) + Spatial Grid Density (64) + Hu Moments (7) + Downsampled Pixels (1024)
    """
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 300:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    pad = 8
    h_img, w_img = mask_u8.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)

    hand_crop = mask_u8[y1:y2, x1:x2]
    norm64 = cv2.resize(hand_crop, (64, 64), interpolation=cv2.INTER_AREA)

    # 1. HOG features (128)
    hog_feats = _compute_custom_hog(norm64)

    # 2. Spatial Grid Density (8x8 = 64 blocks)
    grid8 = cv2.resize(norm64, (8, 8), interpolation=cv2.INTER_AREA).flatten() / 255.0

    # 3. Downsampled silhouette (32x32 = 1024)
    pix32 = (cv2.resize(norm64, (32, 32), interpolation=cv2.INTER_AREA).flatten() / 255.0).astype(np.float32)

    # 4. Hu Moments (7 ค่า)
    moments = cv2.moments(norm64)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return np.hstack([hog_feats, grid8, hu_log, pix32]).astype(np.float32)


def extract_features(bgr_or_rgb: np.ndarray, is_rgb: bool = False) -> tuple[np.ndarray | None, np.ndarray]:
    """รับภาพมือ BGR/RGB -> แปลงเป็น Mask -> ดึง Features"""
    rgb = bgr_or_rgb if is_rgb else cv2.cvtColor(bgr_or_rgb, cv2.COLOR_BGR2RGB)
    mask = extract_hand_mask(rgb)
    feats = extract_features_from_mask(mask)
    return feats, mask


class GestureClassifier:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.classes: list[str] = []
        self.model: SVC | None = None
        if self.model_path.exists():
            self.load()

    def train_from_dataset(self, dataset_dir: Path | str = _ROOT / "dataset") -> dict:
        dataset_path = Path(dataset_dir)
        classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
        if not classes:
            raise FileNotFoundError(f"ไม่พบโฟลเดอร์คลาสใน {dataset_path}")

        X, y = [], []
        print(f"กำลังสกัด Feature จาก Dataset: {classes}")

        for label in classes:
            class_folder = dataset_path / label
            img_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))
            print(f"  - คลาส [{label}]: {len(img_files)} ภาพ...")
            for f in img_files:
                img = cv2.imread(str(f))
                if img is None:
                    continue
                feats, _ = extract_features(img, is_rgb=False)
                if feats is not None:
                    X.append(feats)
                    y.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        print(f"\nรวมข้อมูลที่สกัดได้: {len(X)} ตัวอย่าง, มิติ Feature: {X.shape[1]}")

        # แบ่ง Train 80% / Test 20%
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("กำลังเทรน Support Vector Machine (Linear SVM)...")
        clf = SVC(kernel="linear", probability=True, C=1.0)
        clf.fit(X_train, y_train)

        # ประเมินผล
        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True)
        print(f" ความแม่นยำ (Test Accuracy): {acc * 100:.2f}%\n")

        self.model = clf
        self.classes = classes
        self.save()

        return {"accuracy": acc, "report": report, "samples": len(X)}

    def save(self) -> None:
        if self.model is None:
            return
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self.model, "classes": self.classes}, f)
        print(f"บันทึกโมเดลไว้ที่: {self.model_path}")

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.classes = data["classes"]
        return True

    def predict(self, bgr_crop: np.ndarray) -> tuple[str | None, float, np.ndarray]:
        feats, mask = extract_features(bgr_crop, is_rgb=False)
        if feats is None or self.model is None:
            return None, 0.0, mask

        proba = self.model.predict_proba([feats])[0]
        max_idx = int(np.argmax(proba))
        predicted_class = self.classes[max_idx]
        confidence = float(proba[max_idx])
        return predicted_class, confidence, mask
