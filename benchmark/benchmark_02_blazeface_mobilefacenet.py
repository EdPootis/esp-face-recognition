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
  pip install ai-edge-litert onnxruntime  (Raspberry Pi 3/4, tanpa MediaPipe)

Catatan Raspberry Pi 3/4:
  Wheel MediaPipe terbaru dikompilasi dengan instruksi ARMv8 Crypto (AES),
  sedangkan Cortex-A53 (Pi 3) / Cortex-A72 (Pi 4) milik Broadcom tidak
  memilikinya -> crash "Illegal instruction". Dengan --detector-backend auto
  (default), script mendeteksi hal ini lewat /proc/cpuinfo dan menjalankan
  model BlazeFace .tflite yang sama langsung via TFLite interpreter.

Catatan Jetson Nano:
  MediaPipe 0.8.5 (wheel PINTO0309, Python 3.6) belum punya Tasks API.
  Mode auto otomatis memakai Solutions API (mp.solutions.face_detection).

Cara Menjalankan:
  python benchmark_02_blazeface_mobilefacenet.py --stream-url http://10.7.28.33:81/stream
  python benchmark_02_blazeface_mobilefacenet.py --stream-url http://10.7.28.33:81/stream --no-cuda
  python benchmark_02_blazeface_mobilefacenet.py --detector-backend tflite
===========================================================================
"""

import os
# Harus diset sebelum mediapipe di-import, kalau tidak inisialisasi EGL gagal di Jetson (headless)
os.environ.setdefault('MEDIAPIPE_DISABLE_GPU', '1')
import sys
import time
import glob
import importlib
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
TFLITE_INTERPRETER_MODULES = [
    ('ai_edge_litert.interpreter', 'ai-edge-litert'),
    ('tflite_runtime.interpreter', 'tflite-runtime'),
]


def arm_cpu_lacks_aes():
    """True jika CPU ARM tidak punya instruksi AES (mis. Raspberry Pi 3/4)."""
    if not os.path.exists('/proc/cpuinfo'):
        return False
    try:
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.read().lower().splitlines()
    except Exception:
        return False
    for line in lines:
        # ARM memakai baris "Features", x86 memakai "flags"
        if line.startswith('features'):
            return 'aes' not in line.split(':', 1)[-1].split()
    return False


def resolve_detector_backend(choice):
    """Pilih backend deteksi: 'mediapipe' (Tasks API), 'solutions' (API lama) atau 'tflite'."""
    if choice != 'auto':
        return choice
    if arm_cpu_lacks_aes():
        print("  ⚠️  CPU tidak mendukung instruksi AES (ARMv8 Crypto) -> MediaPipe akan crash")
        print("      (Illegal instruction). Memakai TFLite interpreter langsung.")
        return 'tflite'
    try:
        import mediapipe as mp
    except ImportError:
        return 'mediapipe'  # check_dependencies() yang akan melaporkan
    try:
        importlib.import_module('mediapipe.tasks.python.vision')
        return 'mediapipe'
    except ImportError:
        if hasattr(mp, 'solutions'):
            print("  ⚠️  MediaPipe versi lama tanpa Tasks API -> memakai Solutions API.")
            return 'solutions'
        return 'mediapipe'


def load_tflite_interpreter_class():
    """Cari implementasi TFLite Interpreter yang terpasang."""
    for module_name, _ in TFLITE_INTERPRETER_MODULES:
        try:
            return importlib.import_module(module_name).Interpreter, module_name.split('.')[0]
        except ImportError:
            continue
    try:
        import tensorflow as tf
        return tf.lite.Interpreter, 'tensorflow'
    except ImportError:
        return None, None


def check_dependencies(detector_backend='mediapipe'):
    """Verifikasi dependensi yang diperlukan."""
    missing = []
    if detector_backend in ('mediapipe', 'solutions'):
        try:
            import mediapipe
            # Wheel lama (mis. 0.8.5 untuk Jetson) tidak punya atribut __version__
            print(f"  ✓ MediaPipe {getattr(mediapipe, '__version__', '(versi tidak diketahui)')}")
        except ImportError:
            missing.append("mediapipe")
    else:
        interpreter_cls, source = load_tflite_interpreter_class()
        if interpreter_cls is None:
            missing.append("ai-edge-litert")
        else:
            print(f"  ✓ TFLite Interpreter ({source})")

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
# FACE DETECTION (MediaPipe Solutions API - untuk MediaPipe lama, mis. Jetson 0.8.5)
# ==============================================================================
class BlazeFaceSolutionsDetector:
    """Wrapper deteksi wajah menggunakan mp.solutions.face_detection (BlazeFace short-range)."""

    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        import mediapipe as mp
        face_detection = mp.solutions.face_detection
        try:
            self.detector = face_detection.FaceDetection(
                min_detection_confidence=min_detection_confidence,
                model_selection=model_selection
            )
        except TypeError:
            # Versi yang lebih lama lagi belum punya model_selection (selalu short-range)
            self.detector = face_detection.FaceDetection(
                min_detection_confidence=min_detection_confidence
            )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for det in results.detections:
                # Solutions API memakai koordinat relatif (0.0 - 1.0)
                bbox_rel = det.location_data.relative_bounding_box
                x = max(0, int(bbox_rel.xmin * w))
                y = max(0, int(bbox_rel.ymin * h))
                bw = min(int(bbox_rel.width * w), w - x)
                bh = min(int(bbox_rel.height * h), h - y)

                confidence = det.score[0] if det.score else 0.0

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
# FACE DETECTION (BlazeFace .tflite langsung via TFLite Interpreter)
# ==============================================================================
class BlazeFaceTFLiteDetector:
    """
    Menjalankan model blaze_face_short_range.tflite yang sama dengan MediaPipe,
    tetapi tanpa runtime MediaPipe. Pre/post-processing meniru graph
    MediaPipe face_detection_short_range (letterbox 128x128, SSD anchors,
    sigmoid score, NMS IoU 0.3).
    """

    INPUT_SIZE = 128
    NMS_IOU_THRESHOLD = 0.3

    def __init__(self, min_detection_confidence=0.5, num_threads=4):
        interpreter_cls, self.source = load_tflite_interpreter_class()
        if interpreter_cls is None:
            raise ImportError("TFLite interpreter tidak ditemukan. Jalankan: pip install ai-edge-litert")

        self.min_conf = min_detection_confidence
        self.interpreter = interpreter_cls(model_path=BLAZEFACE_MODEL_PATH, num_threads=num_threads)
        self.interpreter.allocate_tensors()

        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.boxes_index = None
        self.scores_index = None
        for out in self.interpreter.get_output_details():
            if out['shape'][-1] == 16:
                self.boxes_index = out['index']
            elif out['shape'][-1] == 1:
                self.scores_index = out['index']
        if self.boxes_index is None or self.scores_index is None:
            raise RuntimeError("Format output BlazeFace .tflite tidak dikenali.")

        self.anchors = self._generate_anchors(self.INPUT_SIZE)

    @staticmethod
    def _generate_anchors(input_size):
        # Short-range: stride 8 -> 16x16 grid x 2 anchor, stride 16 -> 8x8 grid x 6 anchor = 896
        anchors = []
        for stride, per_cell in ((8, 2), (16, 6)):
            grid = input_size // stride
            for y in range(grid):
                for x in range(grid):
                    for _ in range(per_cell):
                        anchors.append(((x + 0.5) / grid, (y + 0.5) / grid))
        return np.array(anchors, dtype=np.float32)

    def detect(self, frame):
        h, w = frame.shape[:2]
        side = max(h, w)
        pad_x = (side - w) // 2
        pad_y = (side - h) // 2

        # Letterbox ke persegi (seperti MediaPipe), lalu resize ke 128x128
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        square = cv2.copyMakeBorder(rgb, pad_y, side - h - pad_y, pad_x, side - w - pad_x,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        inp = cv2.resize(square, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_AREA)
        inp = (inp.astype(np.float32) / 127.5 - 1.0)[np.newaxis]

        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        raw_boxes = self.interpreter.get_tensor(self.boxes_index).reshape(-1, 16)
        raw_scores = self.interpreter.get_tensor(self.scores_index).reshape(-1)

        scores = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -100.0, 100.0)))
        keep = np.where(scores >= self.min_conf)[0]
        if keep.size == 0:
            return []

        reg = raw_boxes[keep] / self.INPUT_SIZE
        anc = self.anchors[keep]
        scores = scores[keep]

        # Koordinat ternormalisasi (pada gambar persegi) -> piksel frame asli
        cx = (reg[:, 0] + anc[:, 0]) * side - pad_x
        cy = (reg[:, 1] + anc[:, 1]) * side - pad_y
        bw = reg[:, 2] * side
        bh = reg[:, 3] * side
        kp_x = (reg[:, 4::2] + anc[:, 0:1]) * side - pad_x
        kp_y = (reg[:, 5::2] + anc[:, 1:2]) * side - pad_y

        rects = [[int(cx[i] - bw[i] / 2), int(cy[i] - bh[i] / 2), int(bw[i]), int(bh[i])]
                 for i in range(len(keep))]
        nms_idx = cv2.dnn.NMSBoxes(rects, scores.tolist(), self.min_conf, self.NMS_IOU_THRESHOLD)

        faces = []
        for i in np.array(nms_idx).flatten():
            x, y, fw, fh = rects[i]
            x = max(0, x)
            y = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)
            if fw <= 0 or fh <= 0:
                continue
            faces.append({
                "bbox": (x, y, fw, fh),
                "confidence": float(scores[i]),
                "landmarks": [(int(px), int(py)) for px, py in zip(kp_x[i], kp_y[i])]
            })
        return faces

    def release(self):
        self.interpreter = None


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
    det_backend = resolve_detector_backend(args.detector_backend)
    check_dependencies(det_backend)

    device_info = get_device_info()
    device_str = format_device_string(device_info)
    print(f"\n  Device: {device_str}")

    print("\n  📦 Persiapan Model:")
    download_models()

    if det_backend == 'mediapipe':
        print(f"  🔧 Inisialisasi BlazeFace detector (Tasks API)...")
        det = BlazeFaceDetector(min_detection_confidence=0.5, model_selection=0)
        det_label = "MediaPipe BlazeFace (Tasks API)"
        det_runtime = "MediaPipe Tasks API"
        det_input_size = "Automatic (MediaPipe internal)"
    elif det_backend == 'solutions':
        print(f"  🔧 Inisialisasi BlazeFace detector (Solutions API)...")
        det = BlazeFaceSolutionsDetector(min_detection_confidence=0.5, model_selection=0)
        det_label = "MediaPipe BlazeFace (Solutions API)"
        det_runtime = "MediaPipe Solutions API"
        det_input_size = "Automatic (MediaPipe internal)"
    else:
        print(f"  🔧 Inisialisasi BlazeFace detector (TFLite Interpreter langsung)...")
        det = BlazeFaceTFLiteDetector(min_detection_confidence=0.5)
        det_label = f"BlazeFace short-range .tflite ({det.source})"
        det_runtime = f"TFLite Interpreter ({det.source})"
        det_input_size = f"{det.INPUT_SIZE}x{det.INPUT_SIZE} (letterbox)"
    print(f"  ✓ BlazeFace detector siap.")

    print(f"  🔧 Inisialisasi MobileFaceNet recognizer...")
    rec = MobileFaceNetRecognizer(MOBILEFACENET_MODEL_PATH, use_cuda=not args.no_cuda)
    print(f"  ✓ MobileFaceNet recognizer siap. Input: {rec.input_h}x{rec.input_w}")

    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
    active_backend = f"{det_runtime} + ONNX Runtime"

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
        "detection_model": det_label,
        "detection_model_size_bytes": get_model_file_size(BLAZEFACE_MODEL_PATH),
        "recognition_model": os.path.basename(MOBILEFACENET_MODEL_PATH),
        "recognition_model_size_bytes": get_model_file_size(MOBILEFACENET_MODEL_PATH),
        "backend": active_backend,
        "det_input_size": det_input_size,
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
        # Simpan laporan SEBELUM streamer.stop(): di Jetson, FFmpeg bisa abort
        # (Assertion fctx->async_lock failed) saat cap.release() sehingga hasil hilang.
        report = collector.get_full_report()
        report["device_info"] = device_info
        report["model_info"] = model_info
        save_report(report, PAIR_TAG)

        streamer.stop()
        det.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"\n{'=' * 72}")
    print(f"  ✅ BENCHMARK SELESAI: {PAIR_NAME}")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    parser = create_base_argparser(PAIR_NAME)
    parser.add_argument(
        '--detector-backend', choices=['auto', 'mediapipe', 'solutions', 'tflite'], default='auto',
        help="Runtime BlazeFace: 'mediapipe' (Tasks API), 'solutions' (API lama, Jetson\n"
             "MediaPipe 0.8.5), 'tflite' (interpreter langsung, Raspberry Pi 3/4),\n"
             "atau 'auto' (Default: auto)"
    )
    args = parser.parse_args()
    run_benchmark(args)