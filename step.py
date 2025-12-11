import streamlit as st

st.set_page_config(
    page_title="Smart Palm Vision – Dokumentasi Sistem",
    layout="wide"
)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("Algo Team")
section = st.sidebar.radio(
    "Pilih bagian",
    [
        "Ringkasan Sistem",
        "Persiapan Dataset & Training YOLO",
        "Persiapan Lingkungan Python di Windows",
        "Program Deteksi Webcam (detect_sawit.py)",
        "Aplikasi Streamlit Deteksi TBS (streamlit_app.py)",
        "Struktur Proyek & GitHub / QR Code"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Dokumentasi proses pembuatan proyek Smart Palm Vision: Sistem Pendeteksi Kematangan TBS Kelapa Sawit.")

# ==============================
# 1. RINGKASAN SISTEM
# ==============================
if section == "Ringkasan Sistem":
    st.title("Smart Palm Vision")
    st.subheader("Dokumentasi Sistem Pendeteksi Kematangan TBS Sawit")

    st.write(
        "Halaman ini menampilkan dokumentasi lengkap proses pembuatan **Smart Palm Vision**, "
        "yaitu sistem deteksi otomatis tingkat kematangan Tandan Buah Segar (TBS) kelapa sawit "
        "menggunakan model computer vision berbasis YOLO dan antarmuka web Streamlit."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Tujuan Sistem")
        st.write(
            """
            1. Mengidentifikasi tingkat kematangan TBS sawit dari citra kamera.
            2. Mengklasifikasikan TBS ke dalam tiga kategori: Matang, Mengkal, atau Mentah.
            3. Membuat antarmuka web yang interaktif dan mudah digunakan di lapangan.
            4. Menyediakan pencatatan otomatis hasil deteksi untuk setiap pohon.
            """
        )

        st.markdown("#### Komponen Utama")
        st.write(
            """
            1. Dataset TBS sawit (Roboflow).
            2. Model YOLOv8 hasil training di Google Colab.
            3. Program Python lokal dengan deteksi kamera real-time.
            4. Aplikasi Streamlit sebagai antarmuka Smart Palm Vision.
            5. Repository GitHub sebagai dokumentasi digital.
            """
        )

    with col2:
        st.markdown("#### Alur Kerja Sistem Smart Palm Vision")
        st.write(
            """
            1. Menyiapkan dataset dan anotasi gambar TBS sawit.
            2. Melatih model YOLOv8 untuk mendeteksi kematangan.
            3. Membuat program Python untuk uji deteksi kamera.
            4. Membangun aplikasi web Streamlit untuk input foto dan rekapitulasi hasil.
            5. Menyusun dokumentasi langkah-langkah dalam format interaktif.
            """
        )

# ==============================
# 2. DATASET & TRAINING YOLO
# ==============================
elif section == "Persiapan Dataset & Training YOLO":
    st.title("Smart Palm Vision – Dataset & Training YOLO")

    st.markdown("#### 1. Menyiapkan Dataset di Roboflow")
    st.write(
        """
        Dataset Smart Palm Vision terdiri dari gambar TBS sawit dengan tiga label:
        - Matang  
        - Mengkal  
        - Mentah  

        Langkah di Roboflow:
        1. Upload seluruh gambar TBS sawit.
        2. Beri bounding box dan label kelas.
        3. Export dataset dalam format YOLOv8.
        4. Download dataset sebagai file ZIP.
        5. Upload file ZIP ke Google Drive untuk penggunaan Colab.
        """
    )

    st.markdown("#### 2. Menghubungkan Google Drive ke Colab")
    st.code(
        """
from google.colab import drive
drive.mount('/content/drive')
        """,
        language="python"
    )

    st.markdown("#### 3. Meng-unzip Dataset")
    st.code(
        """
!unzip "/content/drive/MyDrive/SMART PALM VISION/sawit_yolo.zip" -d /content/sawit_yolo
!ls /content/sawit_yolo
        """,
        language="bash"
    )

    st.markdown("#### 4. Contoh file data.yaml untuk Smart Palm Vision")
    st.code(
        """
train: /content/sawit_yolo/train/images
val: /content/sawit_yolo/valid/images
test: /content/sawit_yolo/test/images

nc: 3
names:
  - Matang
  - Mengkal
  - Mentah
        """,
        language="yaml"
    )

    st.markdown("#### 5. Instalasi YOLOv8 dan Training Model")
    st.code(
        """
!pip install ultralytics

from ultralytics import YOLO
model = YOLO("yolov8n.pt")

model.train(
    data="/content/sawit_yolo/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8
)
        """,
        language="python"
    )

    st.markdown("#### 6. Mengunduh Model Smart Palm Vision")
    st.code(
        """
!ls runs/detect/train/weights
from google.colab import files
files.download("runs/detect/train/weights/best.pt")
        """,
        language="python"
    )

    st.write("File model diubah nama menjadi **sawit_tbs.pt** untuk digunakan di aplikasi Smart Palm Vision.")

# ==============================
# 3. PERSIAPAN LINGKUNGAN PYTHON WINDOWS
# ==============================
elif section == "Persiapan Lingkungan Python di Windows":
    st.title("Smart Palm Vision – Setup Python Windows")

    st.markdown("#### 1. Struktur Folder Smart Palm Vision")
    st.code(
        """
SmartPalmVision/
├── detect_sawit.py
├── streamlit_app.py
├── smartpalm_steps.py
├── sawit_tbs.pt
└── requirements.txt
        """,
        language="text"
    )

    st.markdown("#### 2. Membuat Virtual Environment")
    st.code(
        """
cd "C:\\Users\\NamaUser\\SmartPalmVision"
python -m venv venv
venv\\Scripts\\activate
        """,
        language="bash"
    )

    st.markdown("#### 3. Instal Library Smart Palm Vision")
    st.code(
        """
pip install streamlit ultralytics opencv-python pillow numpy==1.26.4
        """,
        language="bash"
    )

    st.markdown("#### 4. Uji Instalasi")
    st.code(
        """
python -c "import ultralytics, cv2, numpy; print('Smart Palm Vision Ready')"
        """,
        language="bash"
    )

# ==============================
# 4. DETEKSI WEBCAM
# ==============================
elif section == "Program Deteksi Webcam (detect_sawit.py)":
    st.title("Smart Palm Vision – Program Deteksi Webcam")

    st.markdown("#### File detect_sawit.py")
    st.write("Skrip ini digunakan untuk menguji model YOLO secara langsung.")

    st.code(
        """
import cv2
from ultralytics import YOLO

MODEL_PATH = "sawit_tbs.pt"
CLASS_NAMES = ["Matang", "Mengkal", "Mentah"]

model = YOLO(MODEL_PATH)

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        results = model(frame, imgsz=640, conf=0.4)[0]

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = f"{CLASS_NAMES[cls_id]} ({conf:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

        cv2.imshow("Smart Palm Vision – Webcam Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
        """,
        language="python"
    )

# ==============================
# 5. STREAMLIT APP
# ==============================
elif section == "Aplikasi Streamlit Deteksi TBS (streamlit_app.py)":
    st.title("Smart Palm Vision – Aplikasi Streamlit")

    st.markdown("#### Aplikasi ini memungkinkan:")
    st.write(
        """
        - Pengambilan foto TBS langsung dari kamera.
        - Deteksi tingkat kematangan (Matang/Mengkal/Mentah).
        - Pencatatan hasil per pohon.
        - Reset penomoran pohon.
        """
    )

    st.markdown("#### Struktur Kode streamlit_app.py")
    st.code(
        """
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

MODEL_PATH = "sawit_tbs.pt"
CLASS_NAMES = ["Matang", "Mengkal", "Mentah"]

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

def deteksi_sawit(pil_image):
    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    results = model.predict(img_bgr, imgsz=640, conf=0.4)[0]

    if results.boxes is None:
        return pil_image, None, None

    confs = results.boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))
    box = results.boxes[idx]

    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cls_id = int(box.cls[0].item())
    conf = float(box.conf[0].item())

    label = CLASS_NAMES[cls_id]
    text = f"{label} ({conf:.2f})"

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img_bgr, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    img_rgb_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb_out), label, conf


if "tree_index" not in st.session_state:
    st.session_state.tree_index = 1

if "records" not in st.session_state:
    st.session_state.records = []

st.title("Smart Palm Vision – Deteksi Kematangan TBS Sawit")

img_input = st.camera_input(f"Ambil Foto TBS untuk Pohon {st.session_state.tree_index}")

if st.button("Reset Semua Data"):
    st.session_state.tree_index = 1
    st.session_state.records = []
    st.experimental_rerun()

if img_input:
    pil_img = Image.open(img_input)
    det_img, label, conf = deteksi_sawit(pil_img)
    st.image(det_img, caption="Hasil Deteksi Smart Palm Vision", use_column_width=True)

    if label:
        st.success(f"Pohon {st.session_state.tree_index} = {label} (conf {conf:.2f})")
        if st.button("Simpan Hasil"):
            st.session_state.records.append({
                "Pohon": st.session_state.tree_index,
                "Kematangan": label,
                "Confidence": round(conf, 3)
            })
            st.session_state.tree_index += 1
            st.experimental_rerun()

st.subheader("Rekap Hasil Smart Palm Vision")
st.dataframe(st.session_state.records)
        """,
        language="python"
    )

# ==============================
# 6. GITHUB & QR CODE
# ==============================
elif section == "Struktur Proyek & GitHub / QR Code":
    st.title("Smart Palm Vision – Struktur Proyek & GitHub")

    st.markdown("#### Struktur Folder Final Smart Palm Vision")
    st.code(
        """
SmartPalmVision/
├── detect_sawit.py
├── streamlit_app.py
├── smartpalm_steps.py     # (Dokumentasi interaktif ini)
├── sawit_tbs.pt           # Model YOLO Smart Palm Vision
└── requirements.txt
        """,
        language="text"
    )

    st.markdown("#### Isi requirements.txt")
    st.code(
        """
streamlit
ultralytics
opencv-python
pillow
numpy==1.26.4
        """,
        language="text"
    )

    st.markdown("#### Dokumentasi ke GitHub & QR Code")
    st.write(
        """
        1. Buat repository GitHub bernama **SmartPalmVision**.
        2. Upload semua file proyek ke GitHub.
        3. Upload file **smartpalm_steps.py** ini juga.
        4. Deploy file dokumentasi ini ke Streamlit Cloud (jika ingin versi online).
        5. Ambil link aplikasi Streamlit dan ubah menjadi QR Code.
        6. QR code ditempatkan di laporan atau slide presentasi untuk penguji.
        """
    )

    st.success("Smart Palm Vision documentation is ready for deployment and QR code use.")
