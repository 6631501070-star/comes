"""demo_webcam.py — Live ASL Gesture Recognition & Pipeline Split Screen (OpenCV Window)

รันคำสั่ง:
    conda run -n dl python demo_webcam.py
หรือ:
    python demo_webcam.py

ปุ่มควบคุมบนคีย์บอร์ด (Action Keys):
    - [C] หรือ [X] : ล้างคำทั้งหมด (CLEAR WORD)
    - [SPACE]     : เพิ่มตัวอักษรปัจจุบันลงในคำทันที (Add Letter)
    - [BACKSPACE] : ลบตัวอักษรตัวล่าสุด (Delete)
    - [V]         : สลับมุมมอง แบ่ง 6 ขั้นตอน <-> กล้องเดี่ยว (Toggle View)
    - [Q] / [ESC] : ออกจากโปรแกรม (Quit)
"""

from __future__ import annotations
from pathlib import Path
import time
import cv2
import numpy as np

from module1.spatial_filters import median_filter
from module1.histogram import adjust_contrast_brightness
from module2.segment_skin_hsv import segment_skin_hsv
from module2.apply_morphology import apply_morphology
from module3.classifier import GestureClassifier, extract_features_from_mask

_ROOT = Path(__file__).resolve().parent


def render_tile(
    img: np.ndarray,
    title: str,
    subtitle: str,
    target_w: int = 340,
    target_h: int = 240,
    border_color: tuple[int, int, int] = (60, 60, 60),
) -> np.ndarray:
    """จัดขนาดภาพและใส่แถบ Header ระบุขั้นตอนและโมดูล"""
    if img.ndim == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:
        vis = img.copy()
    else:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = vis.shape[:2]
    header_h = 34
    content_h = target_h - header_h

    scale = min(target_w / w, content_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(vis, (nw, nh), interpolation=cv2.INTER_AREA)

    tile = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    oy = header_h + (content_h - nh) // 2
    ox = (target_w - nw) // 2
    tile[oy : oy + nh, ox : ox + nw] = resized

    # แถบ Header
    cv2.rectangle(tile, (0, 0), (target_w, header_h), (30, 30, 30), -1)
    cv2.putText(
        tile,
        title,
        (8, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        subtitle,
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(tile, (0, 0), (target_w - 1, target_h - 1), border_color, 1)
    return tile


def main() -> None:
    clf = GestureClassifier()
    if clf.model is None:
        print("[INFO] ไม่พบ gesture_model.pkl — กำลังสั่งเทรนโมเดลจาก dataset/ อัตโนมัติ...")
        clf.train_from_dataset()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] ไม่สามารถเปิดกล้องเว็บแคมได้")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ตัวแปรระบบสะกดคำ (Word Builder)
    spelled_word = ""
    last_detected_letter: str | None = None
    letter_hold_start = 0.0
    last_committed_time = 0.0
    auto_commit_duration = 0.8  # ค้างท่า 0.8s พิมพ์ให้อัตโนมัติ
    commit_cooldown = 1.2       # คูลดาวน์ตัวเดิม 1.2s

    show_split_screen = True    # เริ่มต้นที่หน้าจอแบ่ง 6 ขั้นตอน
    window_name = "ComVision — ASL Gesture Recognition & Pipeline Split Screen"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()

    print("=" * 65)
    print(" เปิดกล้องสำเร็จ! ปุ่มกดใช้งาน:")
    print("   [C] หรือ [X]  : ล้างคำทั้งหมด (CLEAR WORD)")
    print("   [SPACE]      : เพิ่มตัวอักษรปัจจุบันทันที (Add Letter)")
    print("   [BACKSPACE]  : ลบตัวอักษรล่าสุด (Delete)")
    print("   [V]          : สลับมุมมอง แบ่ง 6 ขั้นตอน <-> กล้องเดี่ยว")
    print("   [Q] หรือ ESC : ออกจากโปรแกรม")
    print("=" * 65)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            curr_time = time.time()
            fps = 1.0 / max(1e-5, (curr_time - prev_time))
            prev_time = curr_time

            # กรอบวางมือ ROI (280x280)
            rw, rh = 280, 280
            rx = int(w * 0.65) - (rw // 2)
            ry = int(h * 0.5) - (rh // 2)
            rx = max(10, min(rx, w - rw - 10))
            ry = max(10, min(ry, h - rh - 10))

            roi_crop = frame[ry : ry + rh, rx : rx + rw].copy()

            # ---------------- กระบวนการ 6 ขั้นตอน (Data Processing Pipeline) ----------------
            # Step 1: Input Hand (Raw RGB ROI)
            step1_img = roi_crop.copy()
            rgb_in = cv2.cvtColor(step1_img, cv2.COLOR_BGR2RGB)

            # Step 2: Module 1 (Person 1) — Enhancement (Median Denoise + Contrast)
            enhanced_rgb = median_filter(rgb_in, ksize=5)
            enhanced_rgb = adjust_contrast_brightness(enhanced_rgb, alpha=1.15, beta=10)
            step2_img = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

            # Step 3: Module 2 (Person 2) — HSV Skin Segmentation
            limits = {"Hmin": 0.0, "Hmax": 0.12, "Smin": 0.15, "Smax": 0.75, "Vmin": 0.15, "Vmax": 1.0}
            raw_mask, _ = segment_skin_hsv(enhanced_rgb, limits)
            step3_img = (raw_mask.astype(np.uint8) * 255)

            # Step 4: Module 2 (Person 2) — Morphological Cleaning (Open/Close + Fill Holes + Largest Blob)
            morph = apply_morphology(raw_mask, se_shape="disk", se_size=4, keep_largest=True)
            step4_clean = (morph["cleaned_mask"].astype(np.uint8) * 255)

            # Step 5: Module 2 (Person 2) — Canny Edge Detection
            step5_canny = cv2.Canny(step4_clean, 50, 150)

            # Step 6: Module 3 (Person 3) — Geometric Normalization & SVM Prediction
            feats = extract_features_from_mask(step4_clean)
            pred_letter = None
            conf = 0.0

            step6_vis = np.zeros((200, 200, 3), dtype=np.uint8)
            contours, _ = cv2.findContours(step4_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) >= 300:
                    x, y, cw, ch = cv2.boundingRect(largest)
                    pad = 8
                    ih, iw = step4_clean.shape[:2]
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(iw, x + cw + pad), min(ih, y + ch + pad)
                    hand_crop = step4_clean[y1:y2, x1:x2]
                    norm64 = cv2.resize(hand_crop, (64, 64), interpolation=cv2.INTER_AREA)

                    norm_big = cv2.resize(norm64, (120, 120), interpolation=cv2.INTER_NEAREST)
                    step6_vis[40:160, 40:160] = cv2.cvtColor(norm_big, cv2.COLOR_GRAY2BGR)

            if clf.model is not None and feats is not None:
                proba = clf.model.predict_proba([feats])[0]
                max_idx = int(np.argmax(proba))
                pred_letter = clf.classes[max_idx]
                conf = float(proba[max_idx])

            # วาดข้อความผลทำนายใน Step 6
            if pred_letter and conf > 0.45:
                cv2.putText(step6_vis, f"ASL: [{pred_letter}]", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 0), 2)
                cv2.putText(step6_vis, f"Conf: {conf * 100:.1f}%", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            else:
                cv2.putText(step6_vis, "Detecting Hand...", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            # ---------------- ตรรกะสะกดคำ (Word Builder) ----------------
            if pred_letter and conf > 0.55:
                if pred_letter == last_detected_letter:
                    held = curr_time - letter_hold_start
                    if held >= auto_commit_duration and (curr_time - last_committed_time >= commit_cooldown):
                        spelled_word += pred_letter
                        last_committed_time = curr_time
                else:
                    last_detected_letter = pred_letter
                    letter_hold_start = curr_time
            else:
                last_detected_letter = None
                letter_hold_start = curr_time

            # ---------------- การแสดงผลบนหน้าจอ ----------------
            if show_split_screen:
                # สร้างตาราง 6 ช่อง (2 แถว x 3 คอลัมน์)
                tw, th = 340, 230
                t1 = render_tile(step1_img, "Step 1: Input Frame", "Raw RGB Hand ROI", tw, th)
                t2 = render_tile(step2_img, "Step 2: Module 1 (Person 1)", "Enhancement: Median Denoise + Contrast", tw, th)
                t3 = render_tile(step3_img, "Step 3: Module 2 (Person 2)", "Color Segmentation: HSV Skin Mask", tw, th)
                t4 = render_tile(step4_clean, "Step 4: Module 2 (Person 2)", "Morphology: Open/Close + Fill Holes", tw, th)
                t5 = render_tile(step5_canny, "Step 5: Module 2 (Person 2)", "Edge Detection: Canny Contours", tw, th)
                t6 = render_tile(step6_vis, "Step 6: Module 3 (Person 3)", "Geometric 64x64 Normalization & SVM", tw, th, border_color=(0, 200, 100))

                grid = np.vstack([np.hstack([t1, t2, t3]), np.hstack([t4, t5, t6])])

                # แถบล่างแสดงคำสะกด (Word HUD Bar)
                gw = grid.shape[1]
                hud = np.zeros((75, gw, 3), dtype=np.uint8)
                hud[:] = (20, 20, 20)

                # แสดงคำที่สะกด
                w_str = spelled_word if spelled_word else "(empty)"
                curr_str = f" | Current: [{pred_letter}] ({conf * 100:.1f}%)" if pred_letter and conf > 0.4 else " | Current: -"
                cv2.putText(hud, f"WORD: {w_str}{curr_str}", (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255) if spelled_word else (120, 120, 120), 2, cv2.LINE_AA)

                # แสดงปุ่มควบคุมที่ชัดเจน
                keys_info = "[C] or [X]: CLEAR WORD   |   [SPACE]: Add Letter   |   [BACKSPACE]: Delete   |   [V]: Toggle View   |   [Q]: Quit"
                cv2.putText(hud, keys_info, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)

                # FPS มุมขวาบน
                cv2.putText(grid, f"FPS: {fps:.1f}", (gw - 90, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                display = np.vstack([grid, hud])

            else:
                # มุมมองกล้องเดี่ยว
                display = frame.copy()
                roi_col = (0, 255, 0) if pred_letter else (0, 165, 255)
                cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), roi_col, 2)
                cv2.putText(display, "Hand ROI", (rx + 10, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, roi_col, 2)

                if pred_letter and conf > 0.45:
                    txt = f"ASL: [{pred_letter}] ({conf * 100:.1f}%)"
                    cv2.rectangle(display, (rx, ry + rh + 10), (rx + rw, ry + rh + 55), (20, 20, 20), -1)
                    cv2.putText(display, txt, (rx + 15, ry + rh + 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255), 2)

                # แถบล่าง
                cv2.rectangle(display, (0, h - 60), (w, h), (20, 20, 20), -1)
                w_str = spelled_word if spelled_word else "(empty)"
                cv2.putText(display, f"WORD: {w_str}", (20, h - 25), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255) if spelled_word else (120, 120, 120), 2)
                cv2.putText(display, "[C]/[X]: CLEAR   |   [SPACE]: Add   |   [BACKSPACE]: Del   |   [V]: Split View", (w - 580, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)

            cv2.imshow(window_name, display)

            # จัดการปุ่มกด (Key Action)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("c"), ord("C"), ord("x"), ord("X")):
                spelled_word = ""  # CLEAR WORD KEY
                print("[Action] ล้างคำทั้งหมดแล้ว (Word Cleared)")
            elif key == 32:  # SPACE: บังคับเพิ่มตัวอักษร
                if pred_letter:
                    spelled_word += pred_letter
                    print(f"[Action] เพิ่มตัวอักษร: '{pred_letter}' -> คำปัจจุบัน: '{spelled_word}'")
            elif key in (8, 127):  # BACKSPACE: ลบตัวล่าสุด
                if len(spelled_word) > 0:
                    spelled_word = spelled_word[:-1]
                    print(f"[Action] ลบตัวอักษร -> คำปัจจุบัน: '{spelled_word}'")
            elif key in (ord("v"), ord("V")):
                show_split_screen = not show_split_screen

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
