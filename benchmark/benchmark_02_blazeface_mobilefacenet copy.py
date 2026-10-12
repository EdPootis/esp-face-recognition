"""
===========================================================================
  BENCHMARK #2: MediaPipe BlazeFace + MobileFaceNet
===========================================================================
Kombinasi yang didesain untuk edge computing murni.

Face Detection  : MediaPipe BlazeFace (sub-millisecond, dioptimalkan ARM)
Face Recognition: MobileFaceNet (backbone MobileNet, ~1MB model ONNX)
Backend         : MediaPipe (BlazeFace) + ONNX Runtime (MobileFaceNet)

Dependensi tambahan:
  pip install mediapipe onnxruntime  (CPU)
  pip install mediapipe onnxruntime-gpu  (Jetson Nano / CUDA)

Cara Menjalankan:
  python benchmark_02_blazeface_mobilefacenet.py --stream-url http://10.7.28.33:81/stream
  python benchmark_02_blazeface_mobilefacenet.py --stream-url http://10.7.28.33:81/stream --no-cuda
  python benchmark_02_blazeface_mobilefacenet.py --duration 30

Target: Jetson Nano / Raspberry Pi 3
===========================================================================
"""

import os
import sys
import time
import glob
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_utils import (
    CameraStreamer, MetricsCollector, GuidedScenarioRunner,
    GUIDED_SCENARIOS, KNOWN_FACES_DIR, create_base_argparser,
    save_report, get_device_info, format_device_string, get_model_file_size
)

# ==============================================================================
# KONFIGURASI MODEL
# ==============================================================================
PAIR_NAME = "MediaPipe BlazeFace + MobileFaceNet"
PAIR_TAG = "blazeface_mobilefacenet"
DEFAULT_THRESHOLD = 0.55  # Cosine similarity threshold MobileFaceNet

# MobileFaceNet ONNX model
MOBILEFACENET_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'mobilefacenet.onnx')

# Alternatif 1: Model w600k_mbf (MobileFaceNet ArcFace) yang di-hosting di rilis GitHub
MOBILEFACENET_URL = 'https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx'

# Alternatif 2: Repositori HuggingFace publik yang masih aktif menyimpan mobilefacenet.onnx
MOBILEFACENET_URL_ALT = 'https://huggingface.co/Abhishek12233/face-embedding-mobilefacenet-onnx/resolve/main/mobilefacenet.onnx'

# ==============================================================================
# DEPENDENCY CHECK
# ==============================================================================
def check_dependencies():
    """Verifikasi dependensi yang diperlukan."""
    missing = []

    try:
        import mediapipe
        print(f"  ✓ MediaPipe {mediapipe.__version__}")
    except ImportError:
        missing.append("mediapipe")

    try:
        import onnxruntime
        print(f"  ✓ ONNX Runtime {onnxruntime.__version__}")
        providers = onnxruntime.get_available_providers()
        print(f"    Providers: {', '.join(providers)}")
    except ImportError:
        missing.append("onnxruntime")

    if missing:
        print(f"\n  ❌ Dependensi yang belum terpasang: {', '.join(missing)}")
        print(f"  Jalankan: pip install {' '.join(missing)}")
        sys.exit(1)


def download_mobilefacenet():
    """Unduh model MobileFaceNet jika belum ada."""
    if os.path.exists(MOBILEFACENET_MODEL_PATH):
        return

    os.makedirs(os.path.dirname(MOBILEFACENET_MODEL_PATH), exist_ok=True)
    print(f"  📥 Mengunduh MobileFaceNet ONNX model...")

    for url in [MOBILEFACENET_URL, MOBILEFACENET_URL_ALT]:
        try:
            urllib.request.urlretrieve(url, MOBILEFACENET_MODEL_PATH)
            size = os.path.getsize(MOBILEFACENET_MODEL_PATH)
            if size > 100_000:  # Minimal 100KB
                print(f"  ✓ MobileFaceNet berhasil diunduh ({size / 1024:.0f} KB)")
                return
            else:
                os.remove(MOBILEFACENET_MODEL_PATH)
        except Exception as e:
            print(f"  ⚠️  Gagal dari {url}: {e}")

    print(f"  ❌ Gagal mengunduh MobileFaceNet.")
    print(f"     Unduh manual dari salah satu URL berikut ke:")
    print(f"     {MOBILEFACENET_MODEL_PATH}")
    print(f"     URL 1: {MOBILEFACENET_URL}")
    print(f"     URL 2: {MOBILEFACENET_URL_ALT}")
    sys.exit(1)


# ==============================================================================
# FACE DETECTION (MediaPipe BlazeFace)
# ==============================================================================
class BlazeFaceDetector:
    """Wrapper deteksi wajah menggunakan MediaPipe Face Detection (BlazeFace)."""

    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """
        Args:
            min_detection_confidence: Threshold confidence deteksi (0.0 - 1.0)
            model_selection: 0 = short-range (2m), 1 = full-range (5m)
        """
        import mediapipe as mp
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection
        )

    def detect(self, frame):
        """
        Deteksi wajah dalam frame BGR.
        Returns:
            List of dicts: [{"bbox": (x,y,w,h), "confidence": float,
                             "landmarks": [(x,y), ...]}]
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for det in results.detections:
                bbox_rel = det.location_data.relative_bounding_box
                x = int(bbox_rel.xmin * w)
                y = int(bbox_rel.ymin * h)
                bw = int(bbox_rel.width * w)
                bh = int(bbox_rel.height * h)

                # Clamp ke batas frame
                x = max(0, x)
                y = max(0, y)
                bw = min(bw, w - x)
                bh = min(bh, h - y)

                confidence = det.score[0] if det.score else 0.0

                # Extract 6 landmarks (right eye, left eye, nose tip,
                # mouth center, right ear, left ear)
                landmarks = []
                if det.location_data.relative_keypoints:
                    for kp in det.location_data.relative_keypoints:
                        landmarks.append((int(kp.x * w), int(kp.y * h)))

                faces.append({
                    "bbox": (x, y, bw, bh),
                    "confidence": confidence,
                    "landmarks": landmarks
                })

        return faces

    def release(self):
        self.detector.close()


# ==============================================================================
# FACE RECOGNITION (MobileFaceNet via ONNX Runtime)
# ==============================================================================
class MobileFaceNetRecognizer:
    """Ekstraktor fitur wajah menggunakan MobileFaceNet (ONNX Runtime)."""

    def __init__(self, model_path, use_cuda=True):
        import onnxruntime as ort

        providers = []
        if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
            print(f"    ✓ MobileFaceNet: CUDA Execution Provider")
        providers.append('CPUExecutionProvider')

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2

        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # e.g., [1, 3, 112, 112]
        self.output_name = self.session.get_outputs()[0].name

        # Tentukan ukuran input
        if len(self.input_shape) == 4:
            self.input_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 112
            self.input_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 112
        else:
            self.input_h = 112
            self.input_w = 112

        # Warm-up
        dummy = np.random.randn(1, 3, self.input_h, self.input_w).astype(np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})

    def align_and_crop(self, frame, bbox, landmarks=None):
        """
        Crop dan align wajah dari frame berdasarkan bbox.
        Jika ada landmarks, lakukan affine alignment sederhana.
        """
        x, y, w, h = bbox

        # Perlebar crop area sedikit
        margin = 0.2
        mx = int(w * margin)
        my = int(h * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(frame.shape[1], x + w + mx)
        y2 = min(frame.shape[0], y + h + my)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            face_crop = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)

        # Resize ke input model
        face_resized = cv2.resize(face_crop, (self.input_w, self.input_h))
        return face_resized

    def get_embedding(self, face_img):
        """
        Ekstrak embedding vektor dari gambar wajah (BGR, sudah di-crop).
        Returns:
            numpy array 512-dim (atau sesuai model).
        """
        # Preprocessing: BGR -> RGB, normalize, transpose NCHW
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_norm = (face_rgb.astype(np.float32) - 127.5) / 128.0
        face_tensor = np.transpose(face_norm, (2, 0, 1))  # HWC -> CHW
        face_tensor = np.expand_dims(face_tensor, axis=0)  # Add batch dim

        # Inference
        result = self.session.run([self.output_name], {self.input_name: face_tensor})
        embedding = result[0].flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def cosine_similarity(emb1, emb2):
        """Hitung cosine similarity antara dua embedding."""
        return float(np.dot(emb1, emb2))


# ==============================================================================
# ENROLLMENT & RECOGNITION
# ==============================================================================
def load_image_correct_orientation(path, max_dim=800):
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert('RGB')
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def enroll_known_faces(detector, recognizer):
    """Buat database embedding dari folder known_faces."""
    database = []
    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"  ⚠️  Folder '{KNOWN_FACES_DIR}' tidak ditemukan.")
        return database

    print("\n  📂 Enrollment Database Wajah:")
    for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        photos = (glob.glob(os.path.join(person_dir, '*.jpg')) +
                  glob.glob(os.path.join(person_dir, '*.jpeg')) +
                  glob.glob(os.path.join(person_dir, '*.png')))

        count = 0
        for photo in photos:
            try:
                img = load_image_correct_orientation(photo, max_dim=800)
            except Exception:
                continue

            faces = detector.detect(img)
            if not faces:
                continue

            best_face = max(faces, key=lambda f: f["confidence"])
            face_crop = recognizer.align_and_crop(
                img, best_face["bbox"], best_face.get("landmarks")
            )
            embedding = recognizer.get_embedding(face_crop)
            database.append((person_name, embedding))
            count += 1

        print(f"    ✓ '{person_name}': {count} embedding")

    print(f"  Total: {len(database)} embedding dari {len(set(n for n, _ in database))} orang\n")
    return database


def recognize_face(recognizer, embedding, database, threshold):
    """Cocokkan embedding wajah dengan database."""
    best_name = "Unknown"
    best_score = -1.0

    for name, known_emb in database:
        score = recognizer.cosine_similarity(embedding, known_emb)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < threshold:
        return "Unknown", best_score
    return best_name, best_score


# ==============================================================================
# BENCHMARK LOOP
# ==============================================================================
def run_benchmark(args):
    print(f"\n{'=' * 72}")
    print(f"  BENCHMARK: {PAIR_NAME}")
    print(f"{'=' * 72}")

    # 1. Cek dependensi
    print("\n  📦 Verifikasi Dependensi:")
    check_dependencies()

    # 2. Info Device
    device_info = get_device_info()
    device_str = format_device_string(device_info)
    print(f"\n  Device: {device_str}")

    # 3. Download & init models
    print("\n  📦 Persiapan Model:")
    download_mobilefacenet()

    print(f"  🔧 Inisialisasi BlazeFace detector...")
    det = BlazeFaceDetector(min_detection_confidence=0.5, model_selection=0)
    print(f"  ✓ BlazeFace detector siap.")

    print(f"  🔧 Inisialisasi MobileFaceNet recognizer...")
    rec = MobileFaceNetRecognizer(MOBILEFACENET_MODEL_PATH, use_cuda=not args.no_cuda)
    print(f"  ✓ MobileFaceNet recognizer siap. Input: {rec.input_h}x{rec.input_w}")

    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
    active_backend = "MediaPipe + ONNX Runtime"

    # 4. Enroll database
    database = enroll_known_faces(det, rec)

    # 5. Koneksi ke kamera
    print(f"  📷 Menghubungkan ke: {args.stream_url}")
    streamer = CameraStreamer(args.stream_url, timeout=5.0)
    streamer.start()

    print("  ⏳ Menunggu frame pertama...", end="", flush=True)
    wait_start = time.time()
    while time.time() - wait_start < 15.0:
        frame, connected, _ = streamer.get_latest_frame()
        if connected and frame is not None:
            print(f" terhubung! ({frame.shape[1]}x{frame.shape[0]})")
            break
        time.sleep(0.2)
    else:
        print(f"\n  ❌ GAGAL terhubung ke {args.stream_url} dalam 15 detik!")
        streamer.stop()
        return

    # 6. Model info
    model_info = {
        "detection_model": "MediaPipe BlazeFace (Short-Range)",
        "detection_model_size_bytes": "Built-in MediaPipe (~200KB)",
        "recognition_model": os.path.basename(MOBILEFACENET_MODEL_PATH),
        "recognition_model_size_bytes": get_model_file_size(MOBILEFACENET_MODEL_PATH),
        "backend": active_backend,
        "det_input_size": "Automatic (MediaPipe internal)",
        "recognition_threshold": threshold,
        "embedding_dim": "512 (atau sesuai model)",
    }

    # 7. Benchmark guided
    runner = GuidedScenarioRunner()
    collector = MetricsCollector(PAIR_NAME)
    runner.print_header(PAIR_NAME, device_str)

    if not args.no_display:
        window_name = f"Benchmark: {PAIR_NAME}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        for idx, scenario in enumerate(GUIDED_SCENARIOS, 1):
            runner.print_scenario_start(scenario, idx, len(GUIDED_SCENARIOS), args.duration)
            runner.print_countdown(args.countdown)

            collector.begin_scenario(scenario["id"])
            scenario_start = time.time()

            while True:
                elapsed = time.time() - scenario_start
                if elapsed >= args.duration:
                    break

                frame, connected, stream_fps = streamer.get_latest_frame()
                if frame is None or not connected:
                    time.sleep(0.01)
                    continue

                # === DETECTION ===
                t0 = time.perf_counter()
                faces = det.detect(frame)
                t1 = time.perf_counter()
                det_latency = (t1 - t0) * 1000

                face_detected = len(faces) > 0
                rec_latency = 0.0
                det_conf = 0.0
                rec_score = 0.0
                rec_name = "Unknown"
                face_recognized = False

                if face_detected:
                    best_face = max(faces, key=lambda f: f["confidence"])
                    det_conf = best_face["confidence"]

                    # === RECOGNITION ===
                    t2 = time.perf_counter()
                    face_crop = rec.align_and_crop(
                        frame, best_face["bbox"], best_face.get("landmarks")
                    )
                    embedding = rec.get_embedding(face_crop)
                    rec_name, rec_score = recognize_face(rec, embedding, database, threshold)
                    t3 = time.perf_counter()
                    rec_latency = (t3 - t2) * 1000
                    face_recognized = rec_name != "Unknown"

                total_latency = det_latency + rec_latency
                s = collector.scenarios[scenario["id"]]

                collector.record_frame(
                    det_latency_ms=det_latency,
                    rec_latency_ms=rec_latency,
                    total_latency_ms=total_latency,
                    face_detected=face_detected,
                    face_recognized=face_recognized,
                    det_confidence=det_conf,
                    rec_score=rec_score,
                    rec_name=rec_name,
                    stream_fps=stream_fps
                )

                runner.print_progress(
                    elapsed, args.duration,
                    s["frames_processed"],
                    s["faces_detected_count"],
                    s["faces_recognized_count"],
                    stream_fps
                )

                # Display
                if not args.no_display:
                    disp = cv2.resize(frame, (640, 480))
                    fh, fw = frame.shape[:2]
                    sx = 640 / fw
                    sy = 480 / fh

                    for face in faces:
                        x, y, bw, bh = face["bbox"]
                        dx = int(x * sx)
                        dy = int(y * sy)
                        dbw = int(bw * sx)
                        dbh = int(bh * sy)

                        color = (0, 255, 0) if face_recognized else (0, 165, 255)
                        cv2.rectangle(disp, (dx, dy), (dx + dbw, dy + dbh), color, 2)
                        label = f"{rec_name} ({rec_score:.2f})" if face_recognized else f"Det ({det_conf:.2f})"
                        cv2.putText(disp, label, (dx, max(20, dy - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    info_text = f"{PAIR_TAG} | {scenario['id']} | {total_latency:.0f}ms | FPS:{stream_fps:.1f}"
                    cv2.putText(disp, info_text, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.imshow(window_name, disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\n\n  ⏹️  Benchmark dihentikan oleh pengguna.")
                        raise KeyboardInterrupt

                time.sleep(0.005)

            collector.end_scenario()
            summary = collector.get_scenario_summary(scenario["id"])
            runner.print_scenario_result(summary)

    except KeyboardInterrupt:
        collector.end_scenario()
        print("\n  ⏹️  Benchmark dihentikan.")

    finally:
        streamer.stop()
        det.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # 8. Generate & simpan laporan
    report = collector.get_full_report()
    report["device_info"] = device_info
    report["model_info"] = model_info
    save_report(report, PAIR_TAG)

    print(f"\n{'=' * 72}")
    print(f"  ✅ BENCHMARK SELESAI: {PAIR_NAME}")
    print(f"{'=' * 72}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    parser = create_base_argparser(PAIR_NAME)
    args = parser.parse_args()
    run_benchmark(args)
