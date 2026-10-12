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
