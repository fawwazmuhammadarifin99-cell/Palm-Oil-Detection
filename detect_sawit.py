import cv2
from ultralytics import YOLO

# 1. Load model YOLO yang sudah dilatih untuk TBS sawit
MODEL_PATH = "sawit_tbs.pt"   # ganti dengan nama file modelmu
model = YOLO(MODEL_PATH)

# 2. Nama kelas (urutannya harus sama dengan saat training YOLO)
CLASS_NAMES = ["Sawit-Matang"]   # misal cuma 1 kelas

def main():
    # 3. Buka kamera
    cap = cv2.VideoCapture(0)  # 0 = kamera default laptop / USB

    if not cap.isOpened():
        print("Kamera tidak bisa dibuka.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame.")
            break

        # 4. Deteksi dengan YOLO
        # conf=0.4 artinya hanya kotak dengan confidence > 0.4 yang ditampilkan
        results = model(frame, imgsz=640, conf=0.4)[0]

        # 5. Gambar bounding box + label di frame
        if results.boxes is not None:
            for box in results.boxes:
                # koordinat kotak
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # id kelas dan confidence
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # nama kelas
                if 0 <= cls_id < len(CLASS_NAMES):
                    cls_name = CLASS_NAMES[cls_id]
                else:
                    cls_name = "Unknown"

                label = f"{cls_name} ({conf:.2f})"

                # warna biru (BGR)
                color = (255, 0, 0)

                # kotak
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # background tipis untuk teks
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - th - baseline),
                    (x1 + tw, y1),
                    color,
                    thickness=-1,
                )

                # teks (warna putih)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        # 6. Tampilkan hasil
        cv2.imshow("Deteksi TBS Sawit (Matang)", frame)

        # tekan q untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
