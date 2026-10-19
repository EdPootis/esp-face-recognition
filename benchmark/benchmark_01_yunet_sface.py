"""
===========================================================================
  BENCHMARK #1: YuNet + SFace (Ekosistem OpenCV DNN)
===========================================================================
Pasangan model paling ringan dan serbaguna, terintegrasi langsung di OpenCV.

Face Detection  : YuNet (face_detection_yunet_2023mar.onnx, ~227KB)
Face Recognition: SFace (face_recognition_sface_2021dec.onnx, ~37MB)
Backend         : OpenCV cv2.FaceDetectorYN + cv2.FaceRecognizerSF
                  Mendukung CUDA (Jetson Nano) dan CPU (Raspberry Pi)

Opsi --rec-backend ort:
  Inferensi SFace dijalankan lewat ONNX Runtime (CUDAExecutionProvider jika ada),
  sama seperti model FR pada benchmark #2 dan #3. alignCrop() dan match() tetap
  memakai OpenCV. --no-cuda hanya berlaku untuk OpenCV DNN (YuNet / SFace opencv).

Cara Menjalankan:
  python benchmark_01_yunet_sface.py --stream-url http://10.7.28.33:81/stream
  python benchmark_01_yunet_sface.py --stream-url http://10.7.28.33:81/stream --no-cuda
  python benchmark_01_yunet_sface.py --stream-url http://10.7.28.33:81/stream --duration 30
  python benchmark_01_yunet_sface.py --video-file benchmark_video.mp4 --no-cuda --rec-backend ort

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

# Pastikan parent directory ada di sys.path
sys.path.insert(0, os.path.dirname(__file__))
from benchmark_utils import (
    CameraStreamer, VideoFileStreamer, MetricsCollector, GuidedScenarioRunner,
    GUIDED_SCENARIOS, KNOWN_FACES_DIR, create_base_argparser,
    save_report, get_device_info, format_device_string, get_model_file_size
)

# ==============================================================================
# KONFIGURASI MODEL
# ==============================================================================
PAIR_NAME = "YuNet + SFace (OpenCV DNN)"
PAIR_TAG = "yunet_sface"
DEFAULT_THRESHOLD = 0.363  # Cosine similarity threshold SFace

# Path & URL Model
DETECT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'face_detection_yunet_2023mar.onnx')
DETECT_MODEL_URL = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'

RECOG_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 'face_recognition_sface_2021dec.onnx')
RECOG_MODEL_URL = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'

# Salinan SFace untuk ONNX Runtime: bobot yang sama, tetapi 174 initializer dihapus dari
# daftar graph input. Di file asli (exporter lama) bobot ikut terdaftar sebagai input,
# sehingga ONNX Runtime tidak bisa constant folding (±1.7x lebih lambat) dan mencetak
# ratusan warning. Output identik dengan OpenCV (selisih < 1e-5). Dibuat di laptop dengan
# paket `onnx`; jika file tidak ada, file asli tetap dipakai.
RECOG_MODEL_ORT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     'face_recognition_sface_2021dec_ort.onnx')


# ==============================================================================
# FUNGSI MODEL
# ==============================================================================
def download_models():
    """Unduh model jika belum ada."""
    for path, url in [(DETECT_MODEL_PATH, DETECT_MODEL_URL), (RECOG_MODEL_PATH, RECOG_MODEL_URL)]:
        if not os.path.exists(path):
            print(f"  📥 Mengunduh {os.path.basename(path)}...")
            urllib.request.urlretrieve(url, path)
            print(f"  ✓ {os.path.basename(path)} berhasil diunduh.")


def create_face_detector(model_path, det_size, score_th=0.6, nms_th=0.3,
                         top_k=5000, backend_id=None, target_id=None):
    b_id = backend_id if backend_id is not None else getattr(cv2.dnn, 'DNN_BACKEND_DEFAULT', 0)
    t_id = target_id if target_id is not None else getattr(cv2.dnn, 'DNN_TARGET_CPU', 0)

    if hasattr(cv2, 'FaceDetectorYN') and hasattr(cv2.FaceDetectorYN, 'create'):
        return cv2.FaceDetectorYN.create(
            model=model_path, config='', input_size=det_size,
            score_threshold=score_th, nms_threshold=nms_th,
            top_k=top_k, backend_id=b_id, target_id=t_id
        )
    elif hasattr(cv2, 'FaceDetectorYN_create'):
        return cv2.FaceDetectorYN_create(
            model_path, '', det_size, score_th, nms_th, top_k, b_id, t_id
        )
    else:
        raise AttributeError(
            f"OpenCV {cv2.__version__} tidak memiliki modul FaceDetectorYN. "
            f"Diperlukan OpenCV >= 4.5.4."
        )


def create_face_recognizer(model_path, backend_id=None, target_id=None):
    b_id = backend_id if backend_id is not None else getattr(cv2.dnn, 'DNN_BACKEND_DEFAULT', 0)
    t_id = target_id if target_id is not None else getattr(cv2.dnn, 'DNN_TARGET_CPU', 0)

    if hasattr(cv2, 'FaceRecognizerSF') and hasattr(cv2.FaceRecognizerSF, 'create'):
        return cv2.FaceRecognizerSF.create(
            model=model_path, config='', backend_id=b_id, target_id=t_id
        )
    elif hasattr(cv2, 'FaceRecognizerSF_create'):
        return cv2.FaceRecognizerSF_create(model_path, '', b_id, t_id)
    else:
        raise AttributeError(
            f"OpenCV {cv2.__version__} tidak memiliki modul FaceRecognizerSF. "
            f"Diperlukan OpenCV >= 4.5.4."
        )


def init_models(det_size, enable_cuda=True):
    """Inisialisasi YuNet + SFace dengan CUDA auto-fallback."""
    backend_id = getattr(cv2.dnn, 'DNN_BACKEND_DEFAULT', 0)
    target_id = getattr(cv2.dnn, 'DNN_TARGET_CPU', 0)
    active_backend = "CPU (Default)"

    # OpenCV dari pip punya konstanta DNN_BACKEND_CUDA walau tanpa CUDA, dan diam-diam
    # fallback ke CPU tanpa error. Cek device CUDA dulu agar label backend tidak salah.
    try:
        cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception:
        cuda_device_count = 0
    if enable_cuda and cuda_device_count == 0:
        print("  ℹ️  OpenCV tidak mendeteksi device CUDA, memakai CPU.")
        enable_cuda = False

    if enable_cuda:
        cuda_backend = getattr(cv2.dnn, 'DNN_BACKEND_CUDA', None)
        cuda_fp16 = getattr(cv2.dnn, 'DNN_TARGET_CUDA_FP16', None)
        cuda_fp32 = getattr(cv2.dnn, 'DNN_TARGET_CUDA', None)

        if cuda_backend is not None and (cuda_fp16 is not None or cuda_fp32 is not None):
            target_cand = cuda_fp16 if cuda_fp16 is not None else cuda_fp32
            target_name = "CUDA_FP16" if target_cand == cuda_fp16 else "CUDA_FP32"
            try:
                print(f"  🔧 Mencoba inisialisasi CUDA ({target_name})...")
                test_det = create_face_detector(
                    DETECT_MODEL_PATH, det_size, backend_id=cuda_backend, target_id=target_cand
                )
                test_rec = create_face_recognizer(
                    RECOG_MODEL_PATH, backend_id=cuda_backend, target_id=target_cand
                )
                # Warm-up
                dummy = np.zeros((det_size[1], det_size[0], 3), dtype=np.uint8)
                test_det.detect(dummy)
                test_rec.feature(np.zeros((112, 112, 3), dtype=np.uint8))

                backend_id = cuda_backend
                target_id = target_cand
                active_backend = f"GPU CUDA ({target_name})"
                print(f"  ✓ Akselerasi GPU aktif: {target_name}")

                enroll_det = create_face_detector(
                    DETECT_MODEL_PATH, det_size, score_th=0.3,
                    backend_id=backend_id, target_id=target_id
                )
                return test_det, enroll_det, test_rec, active_backend
            except Exception as e:
                print(f"  ⚠️  CUDA gagal ({e}), beralih ke CPU...")

    print("  🔧 Menggunakan CPU Backend.")
    detector = create_face_detector(DETECT_MODEL_PATH, det_size, backend_id=backend_id, target_id=target_id)
    enroll_det = create_face_detector(
        DETECT_MODEL_PATH, det_size, score_th=0.3, backend_id=backend_id, target_id=target_id
    )
    recognizer = create_face_recognizer(RECOG_MODEL_PATH, backend_id=backend_id, target_id=target_id)
    return detector, enroll_det, recognizer, active_backend


class SFaceORTRecognizer:
    """
    SFace dengan inferensi via ONNX Runtime. Antarmuka sama dengan cv2.FaceRecognizerSF:
    alignCrop() dan match() tetap dari OpenCV (CPU, operasi ringan), hanya feature()
    (inferensi model, bagian terberat) yang dijalankan ONNX Runtime.
    """

    def __init__(self, model_path, use_cuda=True):
        import onnxruntime as ort

        providers = []
        if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        ort_model_path = RECOG_MODEL_ORT_PATH if os.path.exists(RECOG_MODEL_ORT_PATH) else model_path
        if ort_model_path == model_path:
            print(f"  ⚠️  {os.path.basename(RECOG_MODEL_ORT_PATH)} tidak ditemukan, memakai model asli "
                  f"(ONNX Runtime lebih lambat karena tidak bisa constant folding).")
        self.model_file = os.path.basename(ort_model_path)

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # sembunyikan warning per-initializer
        self.session = ort.InferenceSession(ort_model_path, sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.provider = self.session.get_providers()[0]
        self._cv = create_face_recognizer(model_path)  # hanya untuk alignCrop() & match()

        # Warm-up (run pertama CUDA jauh lebih lambat karena inisialisasi kernel)
        for _ in range(3):
            self.feature(np.zeros((112, 112, 3), dtype=np.uint8))

    def alignCrop(self, img, face):
        return self._cv.alignCrop(img, face)

    def feature(self, aligned_img):
        # Preprocessing identik dengan FaceRecognizerSF::feature() di OpenCV:
        # blobFromImage(img, scale=1, size=112x112, mean=0, swapRB=true, crop=false)
        blob = cv2.dnn.blobFromImage(aligned_img, 1.0, (112, 112), (0, 0, 0), swapRB=True, crop=False)
        return self.session.run(None, {self.input_name: blob})[0]

    def match(self, feature1, feature2, dis_type):
        return self._cv.match(feature1, feature2, dis_type)


# ==============================================================================
# ENROLLMENT DATABASE WAJAH
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


def enroll_known_faces(enroll_detector, recognizer):
    """Buat database embedding dari folder known_faces."""
    database = []
    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"  ⚠️  Folder '{KNOWN_FACES_DIR}' tidak ditemukan. Database kosong.")
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

            h, w, _ = img.shape
            enroll_detector.setInputSize((w, h))
            _, faces = enroll_detector.detect(img)

            if faces is None or len(faces) == 0:
                continue

            best_face = max(faces, key=lambda f: f[-1])
            aligned = recognizer.alignCrop(img, best_face)
            embedding = recognizer.feature(aligned)
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
        score = recognizer.match(embedding, known_emb, cv2.FaceRecognizerSF_FR_COSINE)
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

    # 1. Info Device
    device_info = get_device_info()
    device_str = format_device_string(device_info)
    print(f"  Device: {device_str}")

    # 2. Download & init models
    print("\n  📦 Persiapan Model:")
    download_models()
    det_size = (args.det_width, args.det_height)
    detector, enroll_det, recognizer, active_backend = init_models(
        det_size, enable_cuda=not args.no_cuda
    )
    if args.rec_backend == 'ort':
        print("  🔧 Memuat SFace via ONNX Runtime...")
        recognizer = SFaceORTRecognizer(RECOG_MODEL_PATH)
        print(f"  ✓ SFace ONNX Runtime aktif: {recognizer.provider} ({recognizer.model_file})")
        active_backend = f"YuNet: OpenCV DNN {active_backend} | SFace: ONNX Runtime ({recognizer.provider})"
    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD

    # 3. Enroll database
    database = enroll_known_faces(enroll_det, recognizer)

    # 4. Koneksi ke kamera ESP32-S3 atau File Video
    if args.video_file:
        print(f"  🎬 Membaca dari file video: {args.video_file}")
        streamer = VideoFileStreamer(args.video_file, override_fps=args.video_fps)
    else:
        print(f"  📷 Menghubungkan ke: {args.stream_url}")
        streamer = CameraStreamer(args.stream_url, timeout=5.0)
        
    streamer.start()

    # Tunggu koneksi
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

    # 5. Info model
    model_info = {
        "detection_model": os.path.basename(DETECT_MODEL_PATH),
        "detection_model_size_bytes": get_model_file_size(DETECT_MODEL_PATH),
        "recognition_model": os.path.basename(RECOG_MODEL_PATH),
        "recognition_model_size_bytes": get_model_file_size(RECOG_MODEL_PATH),
        "backend": active_backend,
        "det_input_size": f"{det_size[0]}x{det_size[1]}",
        "recognition_threshold": threshold,
        "embedding_dim": 128,
    }

    # 6. Mulai benchmark guided
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

                raw_h, raw_w = frame.shape[:2]
                det_w, det_h = det_size

                # === DETECTION ===
                t0 = time.perf_counter()
                det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
                _, faces = detector.detect(det_frame)
                t1 = time.perf_counter()
                det_latency = (t1 - t0) * 1000  # ms

                face_detected = faces is not None and len(faces) > 0
                rec_latency = 0.0
                det_conf = 0.0
                rec_score = 0.0
                rec_name = "Unknown"
                face_recognized = False

                if face_detected:
                    best_face = max(faces, key=lambda f: f[-1])
                    det_conf = float(best_face[-1])

                    # Scale koordinat ke frame asli
                    scale_x = raw_w / det_w
                    scale_y = raw_h / det_h
                    face_scaled = best_face.copy()
                    face_scaled[[0, 2, 4, 6, 8, 10, 12]] *= scale_x
                    face_scaled[[1, 3, 5, 7, 9, 11, 13]] *= scale_y

                    # === RECOGNITION ===
                    t2 = time.perf_counter()
                    aligned = recognizer.alignCrop(frame, face_scaled)
                    embedding = recognizer.feature(aligned)
                    rec_name, rec_score = recognize_face(recognizer, embedding, database, threshold)
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

                # Progress bar di terminal
                runner.print_progress(
                    elapsed, args.duration,
                    s["frames_processed"],
                    s["faces_detected_count"],
                    s["faces_recognized_count"],
                    stream_fps
                )

                # Display frame (opsional)
                if not args.no_display:
                    disp = cv2.resize(frame, (640, 480))
                    if face_detected:
                        scale_d_x = 640 / det_w
                        scale_d_y = 480 / det_h
                        for face in faces:
                            x, y, bw, bh = face[0:4]
                            dx = int(x * scale_d_x)
                            dy = int(y * scale_d_y)
                            dbw = int(bw * scale_d_x)
                            dbh = int(bh * scale_d_y)

                            color = (0, 255, 0) if face_recognized else (0, 165, 255)
                            cv2.rectangle(disp, (dx, dy), (dx + dbw, dy + dbh), color, 2)
                            label = f"{rec_name} ({rec_score:.2f})" if face_recognized else f"Det ({det_conf:.2f})"
                            cv2.putText(disp, label, (dx, max(20, dy - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    # Info overlay
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
        # 7. Generate & simpan laporan SEBELUM streamer.stop(): di Jetson, FFmpeg bisa
        # abort (Assertion fctx->async_lock failed) saat cap.release() sehingga hasil hilang.
        report = collector.get_full_report()
        report["device_info"] = device_info
        report["model_info"] = model_info
        save_report(report, PAIR_TAG)

        streamer.stop()
        if not args.no_display:
            cv2.destroyAllWindows()

    # 8. Print ringkasan akhir
    print(f"\n{'=' * 72}")
    print(f"  ✅ BENCHMARK SELESAI: {PAIR_NAME}")
    print(f"{'=' * 72}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    parser = create_base_argparser(PAIR_NAME)
    parser.add_argument(
        '--rec-backend', choices=['opencv', 'ort'], default='opencv',
        help="Runtime inferensi SFace: 'opencv' (OpenCV DNN) atau 'ort' (ONNX Runtime,\n"
             "CUDA jika tersedia). Default: opencv"
    )
    args = parser.parse_args()
    run_benchmark(args)
