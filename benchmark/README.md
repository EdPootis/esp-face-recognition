# Benchmark Perbandingan Pasangan Model Face Detection + Face Recognition

Suite benchmark untuk membandingkan performa 3 pasangan model AI deteksi & pengenalan wajah
yang ditujukan untuk deployment pada **edge device** (Jetson Nano / Raspberry Pi 3).

Input video berasal dari **ESP32-S3 MJPEG stream** (`http://IP:81/stream`).

---

## Pasangan Model yang Diuji

| # | Pasangan | FD Model | FR Model | Dependensi |
|---|----------|----------|----------|------------|
| 1 | **YuNet + SFace** | YuNet (~227KB) | SFace (~37MB, 128-dim) | OpenCV >= 4.5.4 |
| 2 | **BlazeFace + MobileFaceNet** | MediaPipe BlazeFace | MobileFaceNet ONNX (~1MB) | `mediapipe`, `onnxruntime` |
| 3 | **RetinaFace + ArcFace** | SCRFD-MobileNet (InsightFace) | ArcFace-MBF (512-dim) | `insightface`, `onnxruntime` |

---

## Kombinasi Tambahan (#4 - #7): Tukar Detector & Recognizer

Benchmark #4 - #7 menukar komponen antar pasangan agar terlihat pengaruh detector dan recognizer masing-masing. Semuanya memakai loop, skenario, dan metrik yang sama dengan #1 - #3 (kode bersama di `combo_pipeline.py`).

| # | Script | Detector | Recognizer | Pertanyaan yang dijawab |
|---|--------|----------|------------|-------------------------|
| 4 | `benchmark_04_yolov5face_sface.py` | YOLOv5n-Face / YOLOv5n-0.5-Face (320x256) | SFace | Apakah detector YOLO dengan 5 landmark memperbaiki kelemahan YuNet saat miring kiri? |
| 5 | `benchmark_05_scrfd_sface.py` | SCRFD-500M (detector #3) | SFace (recognizer #1) | Efek recognizer: bandingkan dengan #3 |
| 6 | `benchmark_06_yunet_arcface.py` | YuNet (detector #1) | ArcFace-MBF + alignment (recognizer #2/#3) | Efek recognizer: bandingkan dengan #1 |
| 7 | `benchmark_07_blazefacefull_arcface.py` | BlazeFace **full-range** | ArcFace-MBF + alignment | Uji ulang #2 secara adil (model jarak jauh + alignment) |

Desain ringan untuk Raspberry Pi 3 dan Jetson Nano:
- Tanpa InsightFace, MediaPipe, atau PyTorch. Cukup `opencv-python numpy pillow onnxruntime`, ditambah TFLite interpreter untuk #7 (`ai-edge-litert` di Pi, `tflite_runtime` 2.5 di Jetson).
- Decode SCRFD dan YOLOv5-Face ditulis ulang di numpy. Hasilnya identik dengan implementasi asli (SCRFD vs InsightFace: landmark selisih 0.000 px; embedding ArcFace vs InsightFace: cosine 0.999999).
- Thread spin-wait ONNX Runtime dimatikan agar detector dan recognizer tidak berebut core CPU.
- Kompatibel Python 3.6 (Jetson Nano).

### 1. Siapkan model (sekali, di laptop)

```bash
pip install onnx onnxruntime
python benchmark/prepare_combo_models.py
```

Script ini mengunduh semua model (dicek SHA-256) dan mengonversi YOLOv5-Face: dipotong sebelum layer Detect, input diubah 640x640 -> 320x256 (±2.4x lebih cepat), dan opset 18 -> 13 agar bisa dimuat ONNX Runtime 1.11 di Jetson. Setelah itu salin folder `benchmark/models/` (dan `face_recognition_sface_2021dec_ort.onnx` untuk `--rec-backend ort`) ke perangkat.

### 2. Jalankan

```bash
python benchmark_04_yolov5face_sface.py --video-file benchmark_video.mp4 --no-display
python benchmark_04_yolov5face_sface.py --video-file benchmark_video.mp4 --no-display --yolo-model n-0.5   # paling ringan (Pi 3)
python benchmark_05_scrfd_sface.py      --video-file benchmark_video.mp4 --no-display
python benchmark_06_yunet_arcface.py    --video-file benchmark_video.mp4 --no-display
python benchmark_07_blazefacefull_arcface.py --video-file benchmark_video.mp4 --no-display
```

Argumen tambahan: `--rec-backend opencv|ort` (SFace, #4 dan #5; di Jetson pakai `ort`), `--det-cpu` (detector di CPU walau ada CUDA), `--threads N` (thread ONNX Runtime). YuNet selalu di CPU karena di Jetson lebih cepat (48 ms) daripada OpenCV CUDA (123 ms).

Catatan threshold: ArcFace-MBF memakai 0.45 (sama dengan #3). Benchmark #2 memakai 0.55 untuk model yang **sama**, jadi pakai `--threshold 0.55` di #7 bila ingin membandingkan dengan #2 pada threshold yang sama.

Lisensi: YOLOv5-Face GPL-3.0; model InsightFace (SCRFD, ArcFace-MBF) hanya untuk riset non-komersial; YuNet MIT; SFace dan BlazeFace Apache-2.0.

---

## Cara Menjalankan

### 1. Install Dependensi

```bash
# Dependensi dasar (semua script)
pip install opencv-python numpy Pillow

# Khusus Benchmark #2 (BlazeFace + MobileFaceNet)
pip install mediapipe onnxruntime

# Khusus Benchmark #3 (RetinaFace + ArcFace / InsightFace)
pip install insightface onnxruntime

# Jika menggunakan GPU CUDA (Jetson Nano / Laptop NVIDIA):
pip install onnxruntime-gpu
```

### 2. Jalankan Benchmark (Satu per Satu)

```bash
# Benchmark #1: YuNet + SFace (Paling Ringan)
python benchmark_01_yunet_sface.py --stream-url http://10.7.28.33:81/stream

# Benchmark #2: BlazeFace + MobileFaceNet (Edge Computing)
python benchmark_02_blazeface_mobilefacenet.py --stream-url http://10.7.28.33:81/stream

# Benchmark #3: RetinaFace + ArcFace (Paling Akurat di Antara Model Ringan)
python benchmark_03_retinaface_arcface.py --stream-url http://10.7.28.33:81/stream
```

### 3. Bandingkan Hasil

```bash
python compare_results.py
```

---

## Argumen CLI

| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--stream-url` | `http://10.7.28.33:81/stream` | URL MJPEG stream ESP32-S3 |
| `--duration` | `60` | Durasi per skenario (detik) |
| `--det-width` | `320` | Lebar input deteksi |
| `--det-height` | `240` | Tinggi input deteksi |
| `--no-cuda` | - | Paksa gunakan CPU |
| `--no-display` | - | Mode headless (tanpa jendela OpenCV) |
| `--threshold` | (per model) | Override threshold recognition |
| `--countdown` | `5` | Countdown sebelum setiap skenario |

---

## Skenario Benchmark (Guided)

Setiap script memandu Anda melalui **5 skenario** berurutan:

1. **Wajah Frontal** - Hadap lurus ke kamera (~50-60cm)
2. **Miring Kiri** - Miringkan kepala ~30-45° ke kiri
3. **Miring Kanan** - Miringkan kepala ~30-45° ke kanan
4. **Jarak Jauh** - Mundur ~1.5-2 meter dari kamera
5. **Pencahayaan Redup** - Kurangi cahaya atau berdiri membelakangi sumber cahaya

---

## Metrik yang Dikumpulkan

### Per Skenario:
- **Detection Latency (ms)**: mean, median, P95, P99
- **Recognition Latency (ms)**: mean, median, P95, P99
- **Total Pipeline Latency (ms)**: mean, median, P95, P99
- **Detection Rate (%)**: Persentase frame yang berhasil mendeteksi wajah
- **Recognition Rate (%)**: Persentase wajah terdeteksi yang berhasil dikenali
- **Detection Confidence**: mean, min, max
- **Recognition Score**: mean, min, max
- **Effective FPS**: Frame per detik aktual yang diproses

### Output:
- Terminal print (real-time progress + hasil per skenario)
- File JSON detail di `results/benchmark_<pair>_<timestamp>.json`
- File TXT ringkasan di `results/benchmark_<pair>_<timestamp>.txt`

---

## Struktur Folder

```
benchmark/
├── benchmark_utils.py                  # Modul utilitas bersama
├── benchmark_01_yunet_sface.py         # Benchmark pasangan #1
├── benchmark_02_blazeface_mobilefacenet.py  # Benchmark pasangan #2
├── benchmark_03_retinaface_arcface.py  # Benchmark pasangan #3
├── compare_results.py                  # Analisis perbandingan
├── models/                             # Folder model yang diunduh otomatis
│   └── mobilefacenet.onnx
├── results/                            # Folder hasil benchmark
│   ├── benchmark_yunet_sface_*.json
│   ├── benchmark_yunet_sface_*.txt
│   ├── benchmark_blazeface_mobilefacenet_*.json
│   ├── benchmark_retinaface_arcface_*.json
│   └── comparison_*.json
└── README.md
```
