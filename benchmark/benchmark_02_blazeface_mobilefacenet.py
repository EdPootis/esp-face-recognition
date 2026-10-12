"""
===========================================================================
  BENCHMARK #2: MediaPipe BlazeFace + MobileFaceNet (Tasks API)
===========================================================================
Kombinasi yang didesain untuk edge computing murni.

Face Detection  : MediaPipe BlazeFace (Tasks API, dioptimalkan ARM)
Face Recognition: MobileFaceNet (backbone MobileNet, ~1MB model ONNX)
Backend         : MediaPipe Tasks + ONNX Runtime

Dependensi tambahan:
  pip install mediapipe onnxruntime  (CPU)
  pip install mediapipe onnxruntime-gpu  (Jetson Nano / CUDA)

Cara Menjalankan:
  python benchmark_02_blazeface_mobilefacenet.py --stream-url http://10.7.28.33:81/stream
  python benchmark_02_blazeface_mobilefacenet.py --stream-url http://10.7.28.33:81/stream --no-cuda
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
    CameraStreamer, VideoFileStreamer, MetricsCollector, GuidedScenarioRunner,
    GUIDED_SCENARIOS, KNOWN_FACES_DIR, create_base_argparser,
    save_report, get_device_info, format_device_string, get_model_file_size
)

# ==============================================================================
# KONFIGURASI MODEL
# ==============================================================================
PAIR_NAME = "MediaPipe BlazeFace + MobileFaceNet"
PAIR_TAG = "blazeface_mobilefacenet"
DEFAULT_THRESHOLD = 0.55  # Cosine similarity threshold MobileFaceNet

# Model Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MOBILEFACENET_MODEL_PATH = os.path.join(MODELS_DIR, 'mobilefacenet.onnx')
BLAZEFACE_MODEL_PATH = os.path.join(MODELS_DIR, 'blaze_face_short_range.tflite')

# URLs
MOBILEFACENET_URL = 'https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx'
MOBILEFACENET_URL_ALT = 'https://huggingface.co/Abhishek12233/face-embedding-mobilefacenet-onnx/resolve/main/mobilefacenet.onnx'
BLAZEFACE_URL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite'


# ==============================================================================
# DEPENDENCY CHECK & DOWNLOAD
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


def download_models():
    """Unduh model MobileFaceNet dan BlazeFace (.tflite) jika belum ada."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Download BlazeFace
    if not os.path.exists(BLAZEFACE_MODEL_PATH):
        print(f"  📥 Mengunduh BlazeFace model (MediaPipe Tasks)...")
        try:
            urllib.request.urlretrieve(BLAZEFACE_URL, BLAZEFACE_MODEL_PATH)
            print(f"  ✓ BlazeFace berhasil diunduh.")
        except Exception as e:
            print(f"  ❌ Gagal mengunduh BlazeFace: {e}")
            sys.exit(1)

    # 2. Download MobileFaceNet
    if not os.path.exists(MOBILEFACENET_MODEL_PATH):
        print(f"  📥 Mengunduh MobileFaceNet ONNX model...")
        for url in [MOBILEFACENET_URL, MOBILEFACENET_URL_ALT]:
            try:
                urllib.request.urlretrieve(url, MOBILEFACENET_MODEL_PATH)
                size = os.path.getsize(MOBILEFACENET_MODEL_PATH)
                if size > 100_000:
                    print(f"  ✓ MobileFaceNet berhasil diunduh ({size / 1024:.0f} KB)")
                    return
                else:
                    os.remove(MOBILEFACENET_MODEL_PATH)
            except Exception as e:
                print(f"  ⚠️  Gagal dari {url}: {e}")

        print(f"  ❌ Gagal mengunduh MobileFaceNet.")
        sys.exit(1)


# ==============================================================================
# FACE DETECTION (MediaPipe Tasks API)
# ==============================================================================
class BlazeFaceDetector:
    """Wrapper deteksi wajah menggunakan MediaPipe Tasks API."""

    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp_image = mp.Image
        self.mp_image_format = mp.ImageFormat

        # Inisialisasi menggunakan Tasks API dan model file .tflite
        base_options = python.BaseOptions(model_asset_path=BLAZEFACE_MODEL_PATH)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Konversi ke format Image objek MP
        mp_img = self.mp_image(image_format=self.mp_image_format.SRGB, data=rgb)
        results = self.detector.detect(mp_img)

        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for det in results.detections:
                # Tasks API Bounding Box menggunakan koordinat piksel absolut
                bbox = det.bounding_box
                x = bbox.origin_x
                y = bbox.origin_y
                bw = bbox.width
                bh = bbox.height

                # Clamp ke batas frame
                x = max(0, x)
                y = max(0, y)
                bw = min(bw, w - x)
                bh = min(bh, h - y)

                confidence = det.categories[0].score if det.categories else 0.0

                # Tasks API Keypoints menggunakan rasio relatif (0.0 - 1.0)
                landmarks = []
                if hasattr(det, 'keypoints') and det.keypoints:
                    for kp in det.keypoints:
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
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        if len(self.input_shape) == 4:
            self.input_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 112
            self.input_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 112
        else:
            self.input_h = 112
            self.input_w = 112

        dummy = np.random.randn(1, 3, self.input_h, self.input_w).astype(np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})

    def align_and_crop(self, frame, bbox, landmarks=None):
        x, y, w, h = bbox
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

        face_resized = cv2.resize(face_crop, (self.input_w, self.input_h))
        return face_resized

    def get_embedding(self, face_img):
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_norm = (face_rgb.astype(np.float32) - 127.5) / 128.0
        face_tensor = np.transpose(face_norm, (2, 0, 1))
        face_tensor = np.expand_dims(face_tensor, axis=0)

        result = self.session.run([self.output_name], {self.input_name: face_tensor})
        embedding = result[0].flatten()

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def cosine_similarity(emb1, emb2):
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

    print("\n  📦 Verifikasi Dependensi:")
    check_dependencies()

    device_info = get_device_info()
    device_str = format_device_string(device_info)
    print(f"\n  Device: {device_str}")

    print("\n  📦 Persiapan Model:")
    download_models()

    print(f"  🔧 Inisialisasi BlazeFace detector (Tasks API)...")
    det = BlazeFaceDetector(min_detection_confidence=0.5, model_selection=0)
    print(f"  ✓ BlazeFace detector siap.")

    print(f"  🔧 Inisialisasi MobileFaceNet recognizer...")
    rec = MobileFaceNetRecognizer(MOBILEFACENET_MODEL_PATH, use_cuda=not args.no_cuda)
    print(f"  ✓ MobileFaceNet recognizer siap. Input: {rec.input_h}x{rec.input_w}")

    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
    active_backend = "MediaPipe Tasks API + ONNX Runtime"

    database = enroll_known_faces(det, rec)

    if args.video_file:
        print(f"  🎬 Membaca dari file video: {args.video_file}")
        streamer = VideoFileStreamer(args.video_file, override_fps=args.video_fps)
    else:
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

    model_info = {
        "detection_model": "MediaPipe BlazeFace (Tasks API)",
        "detection_model_size_bytes": get_model_file_size(BLAZEFACE_MODEL_PATH),
        "recognition_model": os.path.basename(MOBILEFACENET_MODEL_PATH),
        "recognition_model_size_bytes": get_model_file_size(MOBILEFACENET_MODEL_PATH),
        "backend": active_backend,
        "det_input_size": "Automatic (MediaPipe internal)",
        "recognition_threshold": threshold,
        "embedding_dim": "512 (atau sesuai model)",
    }

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

    report = collector.get_full_report()
    report["device_info"] = device_info
    report["model_info"] = model_info
    save_report(report, PAIR_TAG)

    print(f"\n{'=' * 72}")
    print(f"  ✅ BENCHMARK SELESAI: {PAIR_NAME}")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    parser = create_base_argparser(PAIR_NAME)
    args = parser.parse_args()
    run_benchmark(args)