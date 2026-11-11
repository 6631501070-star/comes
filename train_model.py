"""train_model.py — สคริปต์สั่งเทรนโมเดล SVM จากโฟลเดอร์ dataset/

วิธีรัน:
    python train_model.py
"""

from module3.classifier import GestureClassifier


def main() -> None:
    print("=" * 60)
    print(" เริ่มกระบวนการเทรน Gesture Recognition Model (Module 3)")
    print("=" * 60)
    clf = GestureClassifier()
    results = clf.train_from_dataset()
    print("=" * 60)
    print(f" เทรนสำเร็จ! ความแม่นยำบน Test Set: {results['accuracy'] * 100:.2f}%")
    print(" โมเดลถูกบันทึกเป็น gesture_model.pkl เรียบร้อยแล้ว")
    print("=" * 60)


if __name__ == "__main__":
    main()

