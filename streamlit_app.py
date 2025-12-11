import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# -------------------------------------------------
# Konfigurasi
# -------------------------------------------------
MODEL_PATH = "sawit_tbs.pt"  # file model hasil training
CLASS_NAMES = ["Matang", "Mengkal", "Mentah"]  # urutan sesuai training
DEFAULT_MAX_TREES = 100

# -------------------------------------------------
# Load model sekali saja (biar tidak lambat)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------------------------------------
# Inisialisasi state
# -------------------------------------------------
if "tree_id" not in st.session_state:
    st.session_state.tree_id = 1

if "records" not in st.session_state:
    st.session_state.records = []

if "max_trees" not in st.session_state:
    st.session_state.max_trees = DEFAULT_MAX_TREES

# -------------------------------------------------
# Fungsi bantu untuk deteksi satu gambar
# -------------------------------------------------
def detect_one_image(frame_bgr):
    """
    Input  : frame BGR (OpenCV)
    Output : (annotated_bgr, cls_name, conf) atau (frame_bgr, None, None) kalau tidak ada deteksi
    """
    results = model(frame_bgr, imgsz=640, conf=0.4)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return frame_bgr, None, None

    # ambil deteksi dengan confidence tertinggi
    boxes = results.boxes
    best_idx = int(boxes.conf.argmax().item())
    box = boxes[best_idx]

    x1, y1, x2, y2 = box.xyxy[0].tolist()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    cls_id = int(box.cls[0].item())
    conf = float(box.conf[0].item())

    if 0 <= cls_id < len(CLASS_NAMES):
        cls_name = CLASS_NAMES[cls_id]
    else:
        cls_name = "Unknown"

    label = f"{cls_name} ({conf:.2f})"

    # gambar kotak dan label di frame
    color = (255, 0, 0)  # biru BGR
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    cv2.rectangle(
        frame_bgr,
        (x1, y1 - th - baseline),
        (x1 + tw, y1),
        color,
        thickness=-1,
    )
    cv2.putText(
        frame_bgr,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    return frame_bgr, cls_name, conf

# -------------------------------------------------
# UI Streamlit
# -------------------------------------------------
st.title("Smart Palm Vision")

st.write(f"Pohon yang sedang dinilai: **{st.session_state.tree_id}**")

max_trees = st.number_input(
    "Maksimum nomor pohon sebelum kembali ke 1",
    min_value=1,
    value=st.session_state.max_trees,
    step=1,
)
st.session_state.max_trees = max_trees

col1, col2 = st.columns(2)

with col1:
    captured = st.camera_input("Ambil foto TBS untuk Pohon ini")

with col2:
    if st.button("Reset ke Pohon 1 dan hapus semua data"):
        st.session_state.tree_id = 1
        st.session_state.records = []
        st.success("Data di-reset.")

# Jika ada gambar dari kamera
if captured is not None:
    # ubah ke OpenCV BGR
    pil_img = Image.open(captured)
    img_rgb = np.array(pil_img)
    frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    annotated_bgr, cls_name, conf = detect_one_image(frame_bgr)

    # tampilkan hasil
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Hasil deteksi", use_column_width=True)

    if cls_name is None:
        st.warning("Tidak ada TBS yang terdeteksi pada gambar ini.")
    else:
        st.success(f"Hasil: Pohon {st.session_state.tree_id} → {cls_name} (conf={conf:.2f})")

        # simpan ke tabel
        st.session_state.records.append(
            {
                "Pohon": st.session_state.tree_id,
                "Kelas": cls_name,
                "Confidence": round(conf, 3),
            }
        )

        # update nomor pohon (wrap kembali ke 1)
        if st.session_state.tree_id >= st.session_state.max_trees:
            st.session_state.tree_id = 1
        else:
            st.session_state.tree_id += 1

# -------------------------------------------------
# Tabel hasil + download
# -------------------------------------------------
st.subheader("Rekap Data Pohon")

if len(st.session_state.records) == 0:
    st.info("Belum ada data. Ambil foto TBS terlebih dahulu.")
else:
    df = pd.DataFrame(st.session_state.records)
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="data_tbs_sawit.csv",
        mime="text/csv",
    )
