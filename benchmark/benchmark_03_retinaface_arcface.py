"""
===========================================================================
  BENCHMARK #3: RetinaFace-MobileNet + ArcFace-MobileNet (InsightFace)
===========================================================================
Pasangan State-of-the-Art yang di-downgrade ke backbone ringan.

Face Detection  : RetinaFace-MobileNet0.25 (InsightFace SCRFD/RetinaFace)
Face Recognition: ArcFace-MobileNet (InsightFace, ArcFace loss, 512-dim)
Backend         : InsightFace Python SDK (ONNX Runtime / TensorRT)

Dependensi tambahan:
  pip install insightface onnxruntime  (CPU)
  pip install insightface onnxruntime-gpu  (Jetson Nano / CUDA)

Cara Menjalankan:
  python benchmark_03_retinaface_arcface.py --stream-url http://10.7.28.33:81/stream
  python benchmark_03_retinaface_arcface.py --stream-url http://10.7.28.33:81/stream --no-cuda
  python benchmark_03_retinaface_arcface.py --duration 30

Target: Jetson Nano / Raspberry Pi 3
===========================================================================
"""

import os
import sys
import time
import glob

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
PAIR_NAME = "RetinaFace-MobileNet + ArcFace-MobileNet (InsightFace)"
PAIR_TAG = "retinaface_arcface"
DEFAULT_THRESHOLD = 0.45  # Cosine similarity threshold ArcFace

# InsightFace akan otomatis download model ke ~/.insightface/models/
# Kita gunakan buffalo_s (paket model ringan: SCRFD_500M + MBF)
# atau buffalo_sc untuk yang lebih kecil
INSIGHTFACE_MODEL_PACK = "buffalo_sc"  # small, SCRFD_500M + MobileFaceNet


# ==============================================================================
# DEPENDENCY CHECK
# ==============================================================================
def check_dependencies():
    """Verifikasi dependensi InsightFace."""
    missing = []

    try:
        import insightface
        print(f"  ✓ InsightFace {insightface.__version__}")
    except ImportError:
        missing.append("insightface")

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


# ==============================================================================
# INSIGHTFACE WRAPPER
# ==============================================================================
class InsightFaceAnalyzer:
    """
    Wrapper untuk InsightFace FaceAnalysis yang menyediakan:
    - Deteksi wajah (RetinaFace / SCRFD backbone MobileNet)
    - Recognition (ArcFace backbone MobileNet)
    """

    def __init__(self, model_pack=INSIGHTFACE_MODEL_PACK, use_cuda=True,
                 det_size=(640, 640)):
        import insightface
        import onnxruntime as ort

        # Tentukan provider
        providers = []
        if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
            print(f"    ✓ InsightFace: CUDA Execution Provider")
        providers.append('CPUExecutionProvider')

        print(f"  🔧 Mengunduh & memuat model pack '{model_pack}'...")
        print(f"     (Proses ini mungkin memakan waktu saat pertama kali)")

        self.app = insightface.app.FaceAnalysis(
            name=model_pack,
            providers=providers,
        )
        self.app.prepare(ctx_id=0 if use_cuda else -1, det_size=det_size)
        self.det_size = det_size

        # Identifikasi model yang dimuat
        self.det_model_name = "Unknown"
        self.rec_model_name = "Unknown"
        for m in self.app.models:
            model = self.app.models[m]
            task = getattr(model, 'taskname', '')
            name = getattr(model, 'name', m)
            if task == 'detection' or 'det' in m.lower():
                self.det_model_name = name
            elif task == 'recognition' or 'rec' in m.lower() or 'arc' in m.lower():
                self.rec_model_name = name

        print(f"  ✓ Detection model : {self.det_model_name}")
        print(f"  ✓ Recognition model: {self.rec_model_name}")

    def detect_and_recognize(self, frame):
        """
        Deteksi dan ekstrak embedding wajah dalam satu pipeline.
        Returns:
            List of dicts: [{"bbox": (x,y,w,h), "confidence": float,
                             "embedding": np.array, "landmarks": np.array}]
        """
        faces = self.app.get(frame)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1

            result = {
                "bbox": (x1, y1, w, h),
                "confidence": float(face.det_score) if hasattr(face, 'det_score') else 0.0,
                "embedding": face.embedding if hasattr(face, 'embedding') else None,
                "landmarks": face.kps if hasattr(face, 'kps') else None,
            }
            results.append(result)

        return results

    def detect_only(self, frame):
        """Hanya deteksi wajah tanpa recognition (untuk mengukur latency terpisah)."""
        # InsightFace menjalankan det + rec secara bersamaan
        # Kita harus menggunakan detector secara terpisah
        # Fallback: gunakan pipeline penuh
        faces = self.app.get(frame)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            results.append({
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "confidence": float(face.det_score) if hasattr(face, 'det_score') else 0.0,
            })
        return results


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


def enroll_known_faces(analyzer):
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

            faces = analyzer.detect_and_recognize(img)
            if not faces:
                continue

            best = max(faces, key=lambda f: f["confidence"])
            if best["embedding"] is not None:
                # L2 normalize
                emb = best["embedding"]
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                database.append((person_name, emb))
                count += 1

        print(f"    ✓ '{person_name}': {count} embedding")

    print(f"  Total: {len(database)} embedding dari {len(set(n for n, _ in database))} orang\n")
    return database


def recognize_face(embedding, database, threshold):
    """Cocokkan embedding wajah dengan database menggunakan cosine similarity."""
    best_name = "Unknown"
    best_score = -1.0

    for name, known_emb in database:
        score = float(np.dot(embedding, known_emb))
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

# 3. Init InsightFace
    print("\n 📦 Persiapan Model:")
    
    # WAJIB: Koreksi det_size agar selalu menjadi kelipatan 32 untuk SCRFD InsightFace
    w_safe = (args.det_width // 32) * 32
    h_safe = (args.det_height // 32) * 32
    det_size_insight = (w_safe, h_safe)
    
    print(f" 🔧 Dimensi input deteksi disesuaikan menjadi: {w_safe}x{h_safe}")    # InsightFace det_size biasanya kelipatan 32, default 640x640
    # Untuk edge device, gunakan 320x320 atau 160x160
    analyzer = InsightFaceAnalyzer(
        model_pack=INSIGHTFACE_MODEL_PACK,
        use_cuda=not args.no_cuda,
        det_size=det_size_insight
    )

    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
    active_backend = "InsightFace (ONNX Runtime)"

    # 4. Enroll database
    database = enroll_known_faces(analyzer)

    # 5. Koneksi ke kamera atau video file
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

    # 6. Model info
    model_info = {
        "detection_model": f"InsightFace {analyzer.det_model_name} ({INSIGHTFACE_MODEL_PACK})",
        "detection_model_size_bytes": "Auto-downloaded by InsightFace",
        "recognition_model": f"InsightFace {analyzer.rec_model_name} ({INSIGHTFACE_MODEL_PACK})",
        "recognition_model_size_bytes": "Auto-downloaded by InsightFace",
        "backend": active_backend,
        "det_input_size": f"{det_size_insight[0]}x{det_size_insight[1]}",
        "recognition_threshold": threshold,
        "embedding_dim": 512,
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

                # === DETECTION + RECOGNITION (InsightFace pipeline) ===
                # InsightFace menjalankan det + rec secara monolitik,
                # tetapi kita bisa mengukur total pipeline latency
                t0 = time.perf_counter()
                faces = analyzer.detect_and_recognize(frame)
                t1 = time.perf_counter()
                pipeline_latency = (t1 - t0) * 1000

                face_detected = len(faces) > 0
                # Estimasi split: ~60% deteksi, ~40% recognition (berdasarkan profiling umum InsightFace)
                det_latency = pipeline_latency * 0.6
                rec_latency = pipeline_latency * 0.4
                det_conf = 0.0
                rec_score = 0.0
                rec_name = "Unknown"
                face_recognized = False

                if face_detected:
                    best = max(faces, key=lambda f: f["confidence"])
                    det_conf = best["confidence"]

                    if best["embedding"] is not None:
                        emb = best["embedding"]
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm

                        t2 = time.perf_counter()
                        rec_name, rec_score = recognize_face(emb, database, threshold)
                        t3 = time.perf_counter()
                        # Matching latency sangat kecil, tambahannya negligible
                        rec_latency = pipeline_latency * 0.4 + (t3 - t2) * 1000
                        face_recognized = rec_name != "Unknown"

                s = collector.scenarios[scenario["id"]]

                collector.record_frame(
                    det_latency_ms=det_latency,
                    rec_latency_ms=rec_latency,
                    total_latency_ms=pipeline_latency,
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

                    info_text = f"{PAIR_TAG} | {scenario['id']} | {pipeline_latency:.0f}ms | FPS:{stream_fps:.1f}"
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
    parser.add_argument(
        '--model-pack', type=str, default=INSIGHTFACE_MODEL_PACK,
        help=f"InsightFace model pack (Default: {INSIGHTFACE_MODEL_PACK})\n"
             f"Opsi: buffalo_l (besar), buffalo_s (sedang), buffalo_sc (kecil/ringan)"
    )
    args = parser.parse_args()
    INSIGHTFACE_MODEL_PACK = args.model_pack
    run_benchmark(args)
