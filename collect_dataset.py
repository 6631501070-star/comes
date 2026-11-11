"""collect_dataset.py — เครื่องมือเก็บภาพมือสำหรับทำ Dataset (ASL Subset) และสั่งเทรนโมเดลใหม่

วิธีใช้งาน:
    conda run -n dl python collect_dataset.py

ปุ่มควบคุมในโปรแกรม:
    - [A], [B], [C], [D], [L], [O], [T] หรือ [1] - [7] : สลับตัวอักษรเป้าหมายทันที
    - [SPACE]     : ถ่าย 1 รูป (บันทึกภาพมือในกรอบ ROI ลง dataset/<LABEL>/)
    - [F]         : เปิด/ปิด โหมดถ่ายรัว (Burst/Continuous Mode)
    - [BACKSPACE] : ลบรูปล่าสุดของคลาสปัจจุบัน (Undo last photo)
    - [ENTER] / [R]: สั่งเทรนโมเดลใหม่จากศูนย์ทันที (Train Model from Scratch)
    - [M]         : เปิด/ปิด หน้าต่าง Preview Mask (Module 2)
    - [Q] / [ESC] : ออกจากโปรแกรม
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import cv2
import numpy as np

# ลอง import module2 และ module3
try:
    from module2 import segment_skin_hsv, apply_morphology
    HAS_MODULE2 = True
except Exception:
    HAS_MODULE2 = False

try:
    from module3.classifier import GestureClassifier
    HAS_CLASSIFIER = True
except Exception:
    HAS_CLASSIFIER = False

# 7 ตัวอักษรเป้าหมาย
SUPPORTED_CLASSES = ["A", "B", "C", "D", "L", "O", "T"]


def get_existing_count(class_dir: Path) -> int:
    """นับจำนวนรูปที่มีอยู่แล้วในโฟลเดอร์ของคลาสนั้นๆ"""
    if not class_dir.exists():
        return 0
    return len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))


def get_next_filename(class_dir: Path, label: str) -> Path:
    """หาชื่อไฟล์ลำดับถัดไป เช่น A_0001.jpg เพื่อไม่ให้เขียนทับไฟล์เดิม"""
    count = 1
    while True:
        target = class_dir / f"{label}_{count:04d}.jpg"
        if not target.exists():
            return target
        count += 1


def delete_last_saved_file(class_dir: Path) -> str | None:
    """ลบไฟล์รูปล่าสุดในโฟลเดอร์คลาส (สำหรับ Undo)"""
    files = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
    if not files:
        return None
    last_file = files[-1]
    last_file.unlink()
    return last_file.name


def draw_hud(
    frame: np.ndarray,
    current_label: str,
    class_counts: dict[str, int],
    roi_coords: tuple[int, int, int, int],
    continuous_mode: bool,
    show_mask: bool,
    status_msg: str,
    status_time: float,
) -> None:
    """วาดหน้าจอแสดงสถานะและข้อมูลการถ่ายบนเฟรม"""
    h, w = frame.shape[:2]
    rx, ry, rw, rh = roi_coords

    # วาดกรอบสี่เหลี่ยม ROI สำหรับวางมือ
    roi_color = (0, 0, 255) if continuous_mode else (0, 255, 0)
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), roi_color, 2)
    cv2.putText(
        frame,
        "Put Hand Here",
        (rx + 10, ry - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        roi_color,
        2,
        cv2.LINE_AA,
    )

    # แถบพื้นหลังด้านบน
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
    # แถบพื้นหลังด้านล่าง
    cv2.rectangle(overlay, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 1) Current Label (ตัวใหญ่เด่นชัด)
    label_text = f"TARGET: [{current_label}]"
    cv2.putText(
        frame,
        label_text,
        (15, 45),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 2) Continuous indicator
    mode_text = "[BURST ON]" if continuous_mode else "[SINGLE]"
    mode_color = (0, 100, 255) if continuous_mode else (180, 180, 180)
    cv2.putText(
        frame,
        mode_text,
        (230, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        mode_color,
        2,
        cv2.LINE_AA,
    )

    # 3) สรุปจำนวนตัวอย่างแต่ละคลาส
    counts_str = " | ".join(f"{c}:{class_counts.get(c, 0)}" for c in SUPPORTED_CLASSES)
    cv2.putText(
        frame,
        counts_str,
        (15, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # 4) ข้อความแจ้งเตือนสถานะล่าสุด (Saved / Trained / Deleted)
    if time.time() - status_time < 3.0 and status_msg:
        cv2.putText(
            frame,
            status_msg,
            (w - 380, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (50, 255, 50),
            2,
            cv2.LINE_AA,
        )

    # 5) แถบคำสั่งด้านล่าง
    help_text = "SPACE: Shot | F: Burst | A-T/1-7: Switch | BACKSPACE: Undo | ENTER/R: Train Model | Q: Quit"
    cv2.putText(
        frame,
        help_text,
        (15, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )


def train_model_from_camera(
    window_name: str,
    frame: np.ndarray,
    out_root: Path,
) -> tuple[bool, str]:
    """แสดงหน้าจอแจ้งเตือนและสั่งเทรนโมเดลใหม่ทั้งหมด"""
    h, w = frame.shape[:2]
    loading = frame.copy()
    cv2.rectangle(loading, (w // 4, h // 3), (3 * w // 4, 2 * h // 3), (25, 25, 25), -1)
    cv2.rectangle(loading, (w // 4, h // 3), (3 * w // 4, 2 * h // 3), (0, 255, 255), 2)
    cv2.putText(
        loading,
        "TRAINING MODEL...",
        (w // 4 + 35, h // 2 - 10),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        loading,
        "Extracting features & fitting SVM (3-5 sec)...",
        (w // 4 + 25, h // 2 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(window_name, loading)
    cv2.waitKey(50)

    try:
        clf = GestureClassifier()
        res = clf.train_from_dataset(dataset_dir=out_root)
        acc_pct = res["accuracy"] * 100
        samples = res["samples"]
        msg = f"Trained: {acc_pct:.1f}% ({samples} imgs)"
        print(f"\n[SUCCESS] เทรนสำเร็จ! ความแม่นยำ: {acc_pct:.2f}% จาก {samples} รูป บันทึกเป็น gesture_model.pkl")
        return True, msg
    except Exception as e:
        print(f"\n[ERROR] เทรนไม่สำเร็จ: {e}")
        return False, f"Train Error: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="ASL Dataset Collector for ComVision")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--out", type=str, default="dataset", help="Output directory")
    parser.add_argument("--roi-size", type=int, default=300, help="Size of square ROI box")
    parser.add_argument("--save-size", type=int, default=224, help="Resized output image size")
    args = parser.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for c in SUPPORTED_CLASSES:
        (out_root / c).mkdir(parents=True, exist_ok=True)

    class_counts = {c: get_existing_count(out_root / c) for c in SUPPORTED_CLASSES}

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] ไม่สามารถเปิดกล้องเบอร์ {args.cam} ได้")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_idx = 0
    current_label = SUPPORTED_CLASSES[current_idx]

    continuous_mode = False
    burst_interval = 0.2
    last_burst_time = 0.0

    show_mask = False
    status_msg = ""
    status_time = 0.0

    window_name = "ComVision — ASL Dataset Collector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("=" * 65)
    print(" ComVision — ASL Dataset Collector & Model Trainer")
    print(f" โฟลเดอร์เก็บข้อมูล: {out_root}")
    print(" ปุ่มลัดหลัก:")
    print("   - SPACE          : ถ่าย 1 รูป")
    print("   - F              : เปิด/ปิด ถ่ายรัวอัตโนมัติ (Burst Mode)")
    print("   - A,B,C,D,L,O,T  : สลับตัวอักษรเป้าหมาย (หรือกด 1-7)")
    print("   - BACKSPACE      : ลบรูปล่าสุด (Undo)")
    print("   - ENTER หรือ R   : สั่งเทรนโมเดลใหม่ทั้งหมดทันที (Retrain SVM Model)")
    print("   - M              : เปิด/ปิด หน้าต่าง Preview Mask")
    print("   - Q หรือ ESC     : ออกจากโปรแกรม")
    print("=" * 65)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rw = min(args.roi_size, w // 2)
            rh = min(args.roi_size, h - 100)
            rx = int(w * 0.65) - (rw // 2)
            ry = int(h * 0.52) - (rh // 2)
            rx = max(10, min(rx, w - rw - 10))
            ry = max(90, min(ry, h - rh - 40))

            roi_crop = frame[ry : ry + rh, rx : rx + rw].copy()

            # บันทึกรูปอัตโนมัติในโหมด Burst
            now = time.time()
            if continuous_mode and (now - last_burst_time >= burst_interval):
                if args.save_size > 0:
                    saved_img = cv2.resize(roi_crop, (args.save_size, args.save_size), interpolation=cv2.INTER_AREA)
                else:
                    saved_img = roi_crop

                target_dir = out_root / current_label
                save_path = get_next_filename(target_dir, current_label)
                cv2.imwrite(str(save_path), saved_img)

                class_counts[current_label] += 1
                status_msg = f"Saved: {save_path.name}"
                status_time = now
                last_burst_time = now

            # Mask Preview
            if show_mask and HAS_MODULE2:
                rgb_roi = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2RGB)
                limits = {"Hmin": 0.0, "Hmax": 0.12, "Smin": 0.15, "Smax": 0.75, "Vmin": 0.15, "Vmax": 1.0}
                mask, _ = segment_skin_hsv(rgb_roi, limits)
                morph = apply_morphology(mask, se_shape="disk", se_size=4, keep_largest=True)
                mask_vis = (morph["cleaned_mask"].astype(np.uint8) * 255)
                mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
                cv2.imshow("Module 2 — Mask Preview", mask_vis)

            display_frame = frame.copy()
            draw_hud(
                display_frame,
                current_label,
                class_counts,
                (rx, ry, rw, rh),
                continuous_mode,
                show_mask,
                status_msg,
                status_time,
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            elif key == 32:  # SPACE: ถ่าย 1 รูป
                if args.save_size > 0:
                    saved_img = cv2.resize(roi_crop, (args.save_size, args.save_size), interpolation=cv2.INTER_AREA)
                else:
                    saved_img = roi_crop

                target_dir = out_root / current_label
                save_path = get_next_filename(target_dir, current_label)
                cv2.imwrite(str(save_path), saved_img)

                class_counts[current_label] += 1
                status_msg = f"Saved: {save_path.name}"
                status_time = time.time()

            elif key in (ord("f"), ord("F")):  # F: Burst mode toggle
                continuous_mode = not continuous_mode
                last_burst_time = 0.0

            elif key in (8, 127):  # BACKSPACE: Undo ลบรูปล่าสุด
                deleted_name = delete_last_saved_file(out_root / current_label)
                if deleted_name:
                    class_counts[current_label] = max(0, class_counts[current_label] - 1)
                    status_msg = f"Deleted: {deleted_name}"
                    status_time = time.time()
                    print(f"[Undo] ลบรูป {deleted_name} เรียบร้อย")

            elif key in (10, 13, ord("r"), ord("R")):  # ENTER หรือ R: สั่งเทรนโมเดลใหม่จากศูนย์
                success, msg = train_model_from_camera(window_name, display_frame, out_root)
                status_msg = msg
                status_time = time.time()

            elif key in (ord("m"), ord("M")):
                show_mask = not show_mask
                if not show_mask:
                    try:
                        cv2.destroyWindow("Module 2 — Mask Preview")
                    except Exception:
                        pass

            # สลับตัวอักษร A, B, C, D, L, O, T
            elif chr(key).upper() in SUPPORTED_CLASSES:
                current_label = chr(key).upper()
                current_idx = SUPPORTED_CLASSES.index(current_label)

            # สลับด้วยตัวเลข 1 - 7
            elif ord("1") <= key <= ord("7"):
                idx = key - ord("1")
                if idx < len(SUPPORTED_CLASSES):
                    current_idx = idx
                    current_label = SUPPORTED_CLASSES[current_idx]

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nสรุปจำนวนตัวอย่างปัจจุบัน:")
        for c in SUPPORTED_CLASSES:
            print(f"  - คลาส [{c}]: {class_counts.get(c, 0)} รูป")


if __name__ == "__main__":
    main()
