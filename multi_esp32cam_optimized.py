import os
import sys
import glob
import time
import json
import base64
import argparse
import threading
import http.client
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageOps

if sys.platform == 'win32':
    import msvcrt
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==============================================================================
# KONFIGURASI KAMERA
# ==============================================================================
# Daftar IP Address ESP32-CAM / ESP32-S3
CAMERA_HOSTS = {
    'ESP32-S3': '10.7.28.33',
    'ESP32-CAM 1': '10.244.226.130',
    'ESP32-CAM 2': '10.244.226.31',
}

def get_camera_urls(mode='stream'):
    cameras = {}
    for name, host in CAMERA_HOSTS.items():
        if mode == 'stream':
            cameras[name] = f"http://{host}:81/stream"
        else:
            cameras[name] = f"http://{host}/capture"
    return cameras

# Database Mahasiswa & Known Faces
KNOWN_FACES_DIR = 'known_faces'
STUDENTS_DB_FILE = 'students_database.json'

_student_cache = {}

def load_student_database():
    global _student_cache
    _student_cache = {}
    if os.path.exists(STUDENTS_DB_FILE):
        try:
            with open(STUDENTS_DB_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                students = data.get('students', {})
                for uid, info in students.items():
                    folder = info.get('folder', '').lower()
                    if folder:
                        _student_cache[folder] = {
                            'uid': uid,
                            'nim': info.get('nim', '-'),
                            'name': info.get('name', folder),
                            'kd_org': info.get('kd_org', info.get('major', '01.00.12.01'))
                        }
            print(f"[DB] Berhasil memuat {len(_student_cache)} data mahasiswa dari {STUDENTS_DB_FILE}")
        except Exception as e:
            print(f"[DB WARN] Gagal membaca {STUDENTS_DB_FILE}: {e}")

# Threshold kemiripan SFace
RECOGNITION_THRESHOLD = 0.363

# Konfigurasi Dashboard Django
DASHBOARD_URL = 'http://10.7.28.21:8000/api/log/'
ENABLE_DASHBOARD = True
DASHBOARD_COOLDOWN_RECOGNIZED = 5.0
DASHBOARD_COOLDOWN_UNKNOWN = 3.0

# Resolusi tile tampilan GUI
TILE_WIDTH = 480
TILE_HEIGHT = 360

# ==============================================================================
# KONFIGURASI OPTIMASI (CUDA & INPUT SIZE RESIZING)
# ==============================================================================
DEFAULT_DET_WIDTH = 320
DEFAULT_DET_HEIGHT = 240

detect_model_path = 'face_detection_yunet_2023mar.onnx'
detect_model_url = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'

recog_model_path = 'face_recognition_sface_2021dec.onnx'
recog_model_url = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'

# Unduh model jika belum ada
for path, dl_url in [(detect_model_path, detect_model_url), (recog_model_path, recog_model_url)]:
    if not os.path.exists(path):
        print(f"Mengunduh model {path}...")
        urllib.request.urlretrieve(dl_url, path)
        print(f"Model {path} berhasil diunduh.")


def create_face_detector(model_path, det_size, score_th=0.6, nms_th=0.3, top_k=5000, backend_id=None, target_id=None):
    b_id = backend_id if backend_id is not None else getattr(cv2.dnn, 'DNN_BACKEND_DEFAULT', 0)
    t_id = target_id if target_id is not None else getattr(cv2.dnn, 'DNN_TARGET_CPU', 0)

    if hasattr(cv2, 'FaceDetectorYN') and hasattr(cv2.FaceDetectorYN, 'create'):
        return cv2.FaceDetectorYN.create(
            model=model_path,
            config='',
            input_size=det_size,
            score_threshold=score_th,
            nms_threshold=nms_th,
            top_k=top_k,
            backend_id=b_id,
            target_id=t_id
        )
    elif hasattr(cv2, 'FaceDetectorYN_create'):
        return cv2.FaceDetectorYN_create(
            model_path, '', det_size, score_th, nms_th, top_k, b_id, t_id
        )
    else:
        raise AttributeError(
            f"OpenCV {cv2.__version__} tidak memiliki modul FaceDetectorYN (YuNet). "
            f"Diperlukan OpenCV versi >= 4.5.4 (Disarankan OpenCV 4.8.0 untuk Jetson Nano)."
        )


def create_face_recognizer(model_path, backend_id=None, target_id=None):
    b_id = backend_id if backend_id is not None else getattr(cv2.dnn, 'DNN_BACKEND_DEFAULT', 0)
    t_id = target_id if target_id is not None else getattr(cv2.dnn, 'DNN_TARGET_CPU', 0)

    if hasattr(cv2, 'FaceRecognizerSF') and hasattr(cv2.FaceRecognizerSF, 'create'):
        return cv2.FaceRecognizerSF.create(
            model=model_path,
            config='',
            backend_id=b_id,
            target_id=t_id
        )
    elif hasattr(cv2, 'FaceRecognizerSF_create'):
        return cv2.FaceRecognizerSF_create(
            model_path, '', b_id, t_id
        )
    else:
        raise AttributeError(
            f"OpenCV {cv2.__version__} tidak memiliki modul FaceRecognizerSF (SFace). "
            f"Diperlukan OpenCV versi >= 4.5.4 (Disarankan OpenCV 4.8.0 untuk Jetson Nano)."
        )


def init_ai_models(det_size=(DEFAULT_DET_WIDTH, DEFAULT_DET_HEIGHT), enable_cuda=True, use_fp16=True):
    # Cek ketersediaan modul FaceDetectorYN & FaceRecognizerSF di OpenCV
    has_yn = (hasattr(cv2, 'FaceDetectorYN') and hasattr(cv2.FaceDetectorYN, 'create')) or hasattr(cv2, 'FaceDetectorYN_create')
    has_sf = (hasattr(cv2, 'FaceRecognizerSF') and hasattr(cv2.FaceRecognizerSF, 'create')) or hasattr(cv2, 'FaceRecognizerSF_create')

    if not has_yn or not has_sf:
        print("\n" + "=" * 70)
        print(f"[ERROR KRITIKAL OPENCV] Versi OpenCV yang terpasang saat ini adalah: {cv2.__version__}")
        print("  -> Modul FaceDetectorYN & FaceRecognizerSF baru tersedia pada OpenCV >= 4.5.4.")
        print("  -> JetPack 4.6 bawaan Jetson Nano membawa OpenCV 4.1.1 (versi terlalu lama).")
        print("  -> Silakan pasang installer OpenCV 4.8.0 CUDA untuk Jetson Nano.")
        print("=" * 70 + "\n")
        sys.exit(1)

    backend_id = getattr(cv2.dnn, 'DNN_BACKEND_DEFAULT', 0)
    target_id = getattr(cv2.dnn, 'DNN_TARGET_CPU', 0)
    active_device = "CPU (Default)"

    cuda_backend = getattr(cv2.dnn, 'DNN_BACKEND_CUDA', None)
    cuda_target_fp16 = getattr(cv2.dnn, 'DNN_TARGET_CUDA_FP16', None)
    cuda_target_fp32 = getattr(cv2.dnn, 'DNN_TARGET_CUDA', None)

    if enable_cuda and cuda_backend is not None and (cuda_target_fp32 is not None or cuda_target_fp16 is not None):
        try:
            target_cand = cuda_target_fp16 if (use_fp16 and cuda_target_fp16 is not None) else cuda_target_fp32
            target_name = "CUDA_FP16" if (target_cand == cuda_target_fp16) else "CUDA_FP32"
            print(f"[AI ENGINE] Mencoba inisialisasi OpenCV CUDA backend ({target_name})...")

            test_detector = create_face_detector(
                model_path=detect_model_path,
                det_size=det_size,
                score_th=0.6,
                nms_th=0.3,
                top_k=5000,
                backend_id=cuda_backend,
                target_id=target_cand
            )
            test_recognizer = create_face_recognizer(
                model_path=recog_model_path,
                backend_id=cuda_backend,
                target_id=target_cand
            )

            dummy_frame = np.zeros((det_size[1], det_size[0], 3), dtype=np.uint8)
            test_detector.detect(dummy_frame)
            dummy_crop = np.zeros((112, 112, 3), dtype=np.uint8)
            test_recognizer.feature(dummy_crop)

            backend_id = cuda_backend
            target_id = target_cand
            active_device = f"GPU CUDA ({target_name})"
            print(f"[OK] [AI ENGINE] Akselerasi GPU aktif: Backend=CUDA, Target={target_name}")

            enroll_det = create_face_detector(
                model_path=detect_model_path,
                det_size=det_size,
                score_th=0.3,
                nms_th=0.3,
                top_k=5000,
                backend_id=backend_id,
                target_id=target_id
            )
            return test_detector, enroll_det, test_recognizer, active_device
        except Exception as e:
            print(f"[AI ENGINE WARN] CUDA tidak tersedia atau gagal diinisialisasi ({e}). Fallback ke CPU...")

    print("[AI ENGINE] Menggunakan CPU Backend (Default).")
    detector = create_face_detector(
        model_path=detect_model_path,
        det_size=det_size,
        score_th=0.6,
        nms_th=0.3,
        top_k=5000,
        backend_id=backend_id,
        target_id=target_id
    )
    enroll_det = create_face_detector(
        model_path=detect_model_path,
        det_size=det_size,
        score_th=0.3,
        nms_th=0.3,
        top_k=5000,
        backend_id=backend_id,
        target_id=target_id
    )
    recog = create_face_recognizer(
        model_path=recog_model_path,
        backend_id=backend_id,
        target_id=target_id
    )
    return detector, enroll_det, recog, active_device


# ==============================================================================
# CLASS MULTITHREADING STREAMER
# ==============================================================================
class CameraStreamer(threading.Thread):
    def __init__(self, cam_name, url, mode='stream', timeout=5.0):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.url = url
        self.mode = mode.lower()
        self.timeout = timeout
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()

        parsed = urllib.parse.urlparse(self.url)
        self.host = parsed.netloc
        self.path = parsed.path if parsed.path else ('/stream' if self.mode == 'stream' else '/capture')
        if parsed.query:
            self.path += '?' + parsed.query
        self.conn = None

    def _get_connection(self):
        if self.conn is None:
            self.conn = http.client.HTTPConnection(self.host, timeout=self.timeout)
        return self.conn

    def _run_stream(self):
        req = urllib.request.Request(self.url, headers={'User-Agent': 'ESP32CAM-Client'})
        with urllib.request.urlopen(req, timeout=self.timeout) as stream:
            bytes_buffer = b''
            while self.running:
                chunk = stream.read(8192)
                if not chunk:
                    break
                bytes_buffer += chunk

                a = bytes_buffer.find(b'\xff\xd8')
                b = bytes_buffer.find(b'\xff\xd9')

                if a != -1 and b != -1 and b > a:
                    last_b = bytes_buffer.rfind(b'\xff\xd9')
                    last_a = bytes_buffer.rfind(b'\xff\xd8', 0, last_b)

                    if last_a != -1 and last_b != -1 and last_b > last_a:
                        jpg_data = bytes_buffer[last_a:last_b + 2]
                        bytes_buffer = bytes_buffer[last_b + 2:]
                    else:
                        jpg_data = bytes_buffer[a:b + 2]
                        bytes_buffer = bytes_buffer[b + 2:]

                    decoded = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if decoded is not None:
                        with self.lock:
                            self.frame = decoded
                            self.connected = True
                        self._frame_count += 1

                        now = time.time()
                        if now - self._fps_timer >= 1.0:
                            self.fps = self._frame_count / (now - self._fps_timer)
                            self._frame_count = 0
                            self._fps_timer = now
                    else:
                        with self.lock:
                            self.connected = False

    def _run_capture(self):
        conn = self._get_connection()
        conn.request('GET', self.path, headers={'User-Agent': 'ESP32CAM-Client', 'Connection': 'keep-alive'})
        resp = conn.getresponse()

        if resp.status == 200:
            raw_data = resp.read()
            img_np = np.frombuffer(raw_data, dtype=np.uint8)
            decoded = cv2.imdecode(img_np, -1)

            if decoded is not None:
                with self.lock:
                    self.frame = decoded
                    self.connected = True
                self._frame_count += 1

                now = time.time()
                if now - self._fps_timer >= 1.0:
                    self.fps = self._frame_count / (now - self._fps_timer)
                    self._frame_count = 0
                    self._fps_timer = now
            else:
                with self.lock:
                    self.connected = False
        else:
            resp.read()
            with self.lock:
                self.connected = False

        time.sleep(0.005)

    def run(self):
        while self.running:
            try:
                if self.mode == 'stream':
                    self._run_stream()
                else:
                    self._run_capture()
            except Exception:
                with self.lock:
                    self.connected = False
                if self.conn:
                    try:
                        self.conn.close()
                    except Exception:
                        pass
                    self.conn = None
                time.sleep(0.5)

    def get_latest_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.connected, self.fps
            return None, self.connected, self.fps

    def stop(self):
        self.running = False
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass


# ==============================================================================
# FUNGSI EMBEDDING & FACE RECOGNITION
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


def get_embedding(recognizer, image, face_row):
    aligned_face = recognizer.alignCrop(image, face_row)
    return recognizer.feature(aligned_face)


def crop_face_roi(image, face_row, margin=0.25):
    h, w = image.shape[:2]
    fx, fy, fw, fh = map(int, face_row[:4])
    mx = int(fw * margin)
    my = int(fh * margin)
    x1 = max(0, fx - mx)
    y1 = max(0, fy - my)
    x2 = min(w, fx + fw + mx)
    y2 = min(h, fy + fh + my)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image
    return crop


def enroll_known_faces(enroll_detector, recognizer):
    database = []
    if not os.path.isdir(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        return database

    print("\n--- Memulai Enrollment Database Wajah ---")
    for person_name in sorted(os.listdir(KNOWN_FACES_DIR)):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        photo_paths = glob.glob(os.path.join(person_dir, '*.jpg')) + \
                      glob.glob(os.path.join(person_dir, '*.jpeg')) + \
                      glob.glob(os.path.join(person_dir, '*.png'))

        count = 0
        for photo_path in photo_paths:
            try:
                img = load_image_correct_orientation(photo_path, max_dim=800)
            except Exception:
                continue

            h, w, _ = img.shape
            enroll_detector.setInputSize((w, h))
            _, faces = enroll_detector.detect(img)

            if faces is None or len(faces) == 0:
                continue

            best_face = max(faces, key=lambda f: f[-1])
            embedding = get_embedding(recognizer, img, best_face)
            database.append((person_name, embedding))
            count += 1

        print(f"  [OK] '{person_name}': {count} embedding ter-enroll.")

    return database


def recognize(recognizer, embedding, database):
    best_name = "Unknown"
    best_score = -1.0
    for name, known_embedding in database:
        score = recognizer.match(embedding, known_embedding, cv2.FaceRecognizerSF_FR_COSINE)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < RECOGNITION_THRESHOLD:
        return "Unknown", best_score
    return best_name, best_score


_cam_last_notif = {}
_dashboard_last_sent = {}


def send_face_to_esp32_async(cam_url, name, score, cam_name):
    def _worker():
        try:
            if not cam_url:
                return
            parsed = urllib.parse.urlparse(cam_url)
            host = parsed.hostname or parsed.netloc.split(':')[0]
            scheme = parsed.scheme or 'http'
            base_url = f"{scheme}://{host}"

            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            encoded_name = urllib.parse.quote(name)
            encoded_cam = urllib.parse.quote(cam_name)
            encoded_time = urllib.parse.quote(time_str)

            target_url = (f"{base_url}/face?"
                          f"name={encoded_name}&"
                          f"score={score:.2f}&"
                          f"cam={encoded_cam}&"
                          f"time={encoded_time}")

            req = urllib.request.Request(target_url, headers={'User-Agent': 'FaceRecClient'})
            with urllib.request.urlopen(req, timeout=1.5) as _:
                pass
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


def send_log_to_dashboard_async(dashboard_url, cam_name, name, score, face_status, face_crop=None):
    def _worker():
        try:
            foto_b64 = None
            if face_crop is not None:
                success, encoded_img = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if success:
                    raw_b64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
                    foto_b64 = f"data:image/jpeg;base64,{raw_b64}"

            now_dt = datetime.now()
            mac_addr = CAMERA_HOSTS.get(cam_name, '00:00:00:00:00:00')
            st_info = _student_cache.get(name.lower(), {})
            nim = st_info.get('nim', '-')
            uid = st_info.get('uid', '-')
            kd_org = st_info.get('kd_org', '01.00.12.01')
            full_name = st_info.get('name', name)

            payload = {
                'no_identitas': nim,
                'nama': full_name,
                'rfid_tag': uid,
                'mac_address': mac_addr,
                'kd_org': kd_org,
                'tipe_perangkat': 'ESP32-CAM Face Recognition',
                'ts_presensi_device': now_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'kamera': cam_name,
                'skor': round(float(score), 4),
                'status': 'HADIR' if face_status == 'recognized' else 'UNKNOWN',
                'foto_b64': foto_b64,
                'nim': nim,
                'person_name': full_name,
                'similarity': round(float(score), 4),
                'recognized': (face_status == 'recognized'),
                'timestamp': now_dt.isoformat(),
                'image_base64': foto_b64
            }
            data_bytes = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                dashboard_url,
                data=data_bytes,
                headers={'Content-Type': 'application/json', 'User-Agent': 'FaceRecClient'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                status_code = resp.getcode()
                print(f"[DASHBOARD] Log terkirim ({status_code}) -> {cam_name} | {full_name} ({nim}) | Skor: {score:.2f}")
        except Exception as e:
            print(f"[DASHBOARD ERROR] Gagal mengirim log ({cam_name} / {name}): {e}")

    threading.Thread(target=_worker, daemon=True).start()


# ==============================================================================
# PEMROSESAN & TAMPILAN GRID DENGAN INPUT SIZE TEROPTIMASI
# ==============================================================================
def process_camera_frame(raw_frame, cam_name, cam_url, connected, fps, known_faces_db, face_detector, recognizer, det_size, active_device="CPU", render_display=True):
    if not connected or raw_frame is None:
        if not render_display:
            return None
        tile = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
        tile[:] = (30, 30, 30)
        cv2.putText(tile, cam_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(tile, "[ OFFLINE / RECONNECTING ]", (20, TILE_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return tile

    raw_h, raw_w = raw_frame.shape[:2]
    det_w, det_h = det_size

    # OPTIMASI: Rescale frame ke input size deteksi (320x240)
    det_frame = cv2.resize(raw_frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
    _, faces = face_detector.detect(det_frame)

    display_frame = None
    if render_display:
        display_frame = cv2.resize(raw_frame, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)

    scale_raw_x = raw_w / det_w
    scale_raw_y = raw_h / det_h
    scale_disp_x = TILE_WIDTH / det_w
    scale_disp_y = TILE_HEIGHT / det_h

    if faces is not None and len(faces) > 0:
        global _cam_last_notif
        for face in faces:
            # Rescale koordinat 15-elemen (bbox + 5 facial landmarks) kembali ke raw_frame
            face_scaled = face.copy()
            face_scaled[[0, 2, 4, 6, 8, 10, 12]] *= scale_raw_x
            face_scaled[[1, 3, 5, 7, 9, 11, 13]] *= scale_raw_y

            embedding = get_embedding(recognizer, raw_frame, face_scaled)
            name, score = recognize(recognizer, embedding, known_faces_db)

            # Kirim notifikasi ke ESP32 LCD
            now = time.time()
            last_name, last_time = _cam_last_notif.get(cam_name, ("", 0))
            if (now - last_time > 1.5) or (name != last_name):
                send_face_to_esp32_async(cam_url, name, score, cam_name)
                _cam_last_notif[cam_name] = (name, now)

            # Kirim log absensi ke Dashboard Django
            if ENABLE_DASHBOARD:
                global _dashboard_last_sent
                face_status = "recognized" if name != "Unknown" else "unrecognized"
                cooldown = DASHBOARD_COOLDOWN_RECOGNIZED if face_status == "recognized" else DASHBOARD_COOLDOWN_UNKNOWN
                dash_key = (cam_name, name if face_status == "recognized" else "Unknown")
                last_dash_time = _dashboard_last_sent.get(dash_key, 0)

                if now - last_dash_time > cooldown:
                    face_crop = crop_face_roi(raw_frame, face_scaled)
                    send_log_to_dashboard_async(DASHBOARD_URL, cam_name, name, score, face_status, face_crop)
                    _dashboard_last_sent[dash_key] = now

            if render_display and display_frame is not None:
                # Koordinat bounding box pada display window
                x, y, bw, bh = face[0:4]
                dx = int(x * scale_disp_x)
                dy = int(y * scale_disp_y)
                dw = int(bw * scale_disp_x)
                dh = int(bh * scale_disp_y)

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (dx, dy), (dx + dw, dy + dh), color, 2)
                label = f"{name} ({score:.2f})"
                cv2.putText(display_frame, label, (dx, max(15, dy - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if render_display and display_frame is not None:
        # Header status kamera (Nama + FPS + Engine)
        cv2.rectangle(display_frame, (0, 0), (TILE_WIDTH, 28), (0, 0, 0), -1)
        cv2.circle(display_frame, (12, 14), 5, (0, 255, 0), -1)
        status_text = f"{cam_name} | {fps:.1f} FPS | {active_device}"
        cv2.putText(display_frame, status_text, (25, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    return display_frame


def create_grid_display(tiles):
    n = len(tiles)
    if n == 0:
        return np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
    if n == 1:
        return tiles[0]
    if n == 2:
        return np.hstack((tiles[0], tiles[1]))

    cols = 2
    rows = (n + cols - 1) // cols

    padded_tiles = list(tiles)
    while len(padded_tiles) < rows * cols:
        blank = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
        blank[:] = (20, 20, 20)
        cv2.putText(blank, "No Camera", (TILE_WIDTH // 3, TILE_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        padded_tiles.append(blank)

    grid_rows = []
    for r in range(rows):
        row_tiles = padded_tiles[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_tiles))

    return np.vstack(grid_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ESP32-CAM Multi-Stream Face Recognition (OpenCV CUDA & Downscaled Input Size)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'mode_pos',
        nargs='?',
        choices=['stream', 'capture', 's', 'c'],
        default=None,
        help="Mode endpoint kamera:\n  'stream'  : Menggunakan MJPEG stream (port 81 /stream)\n  'capture' : Menggunakan snapshot polling Keep-Alive (port 80 /capture)\n  (Default: stream)"
    )
    parser.add_argument(
        '-m', '--mode',
        choices=['stream', 'capture', 's', 'c'],
        default=None,
        dest='mode_opt',
        help="Mode endpoint kamera: 'stream' atau 'capture' (Default: stream)"
    )
    parser.add_argument('--no-cuda', action='store_true', help="Paksa gunakan CPU (nonaktifkan CUDA GPU)")
    parser.add_argument('--no-fp16', action='store_true', help="Gunakan FP32 alih-alih FP16 saat CUDA aktif")
    parser.add_argument('--det-width', type=int, default=DEFAULT_DET_WIDTH, help=f"Lebar input deteksi (Default: {DEFAULT_DET_WIDTH})")
    parser.add_argument('--det-height', type=int, default=DEFAULT_DET_HEIGHT, help=f"Tinggi input deteksi (Default: {DEFAULT_DET_HEIGHT})")
    parser.add_argument('--headless', action='store_true', help="Jalankan mode headless tanpa GUI OpenCV (sangat direkomendasikan untuk Jetson Nano via SSH)")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_mode = args.mode_opt or args.mode_pos or 'stream'
    mode = 'stream' if raw_mode.lower() in ['s', 'stream'] else 'capture'

    det_size = (args.det_width, args.det_height)
    enable_cuda = not args.no_cuda
    use_fp16 = not args.no_fp16

    # Deteksi otomatis mode headless jika DISPLAY tidak terdeteksi (SSH Linux / Jetson Nano)
    is_headless = args.headless or (sys.platform != 'win32' and os.environ.get('DISPLAY') is None)

    print("=" * 68)
    print("ESP32-CAM Multi-Stream Face Recognition System (TEROPTIMASI)")
    print(f"Mode Endpoint    : {mode.upper()} ({'/stream (Port 81)' if mode == 'stream' else '/capture (Port 80)'})")
    print(f"Detection Res    : {det_size[0]}x{det_size[1]} (Downscaled Input Size)")
    print(f"Akselerasi CUDA  : {'Aktif (Auto/Preferred)' if enable_cuda else 'Nonaktif (Forced CPU)'}")
    print(f"Mode Display     : {'HEADLESS (Tanpa GUI Display - Hemat CPU/RAM)' if is_headless else 'GUI WINDOW (OpenCV HighGUI)'}")
    print("=" * 68)

    # 1. Inisialisasi Model AI (YuNet + SFace)
    face_detector, enroll_detector, recognizer, active_device = init_ai_models(
        det_size=det_size,
        enable_cuda=enable_cuda,
        use_fp16=use_fp16
    )

    # 2. Pendaftaran Database Wajah & Profil Mahasiswa
    load_student_database()
    known_faces_db = enroll_known_faces(enroll_detector, recognizer)
    total_people = len(set(n for n, _ in known_faces_db))
    print(f"Total database: {len(known_faces_db)} foto dari {total_people} profil terdaftar.\n")

    # 3. Inisialisasi & Start Thread tiap Kamera
    cameras = get_camera_urls(mode)
    streamers = []
    print(f"Memulai background thread untuk setiap kamera (Mode: {mode.upper()}):")
    for cam_name, url in cameras.items():
        print(f" -> Menghubungkan ke [{cam_name}]: {url}")
        streamer = CameraStreamer(cam_name, url, mode=mode)
        streamer.start()
        streamers.append(streamer)

    if not is_headless:
        print(f"\nSemua kamera telah diaktifkan ({active_device}).")
        print("Tekan 'q' pada jendela GUI atau di terminal untuk keluar.\n")
        window_name = f"Multi-Camera ESP32 Face Recognition [{active_device}]"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    else:
        print(f"\nSemua kamera telah diaktifkan ({active_device}) dalam mode HEADLESS.")
        print("Tekan Ctrl+C di terminal untuk keluar.\n")

    try:
        while True:
            if sys.platform == 'win32' and msvcrt.kbhit():
                if msvcrt.getch().lower() == b'q':
                    print("\nTombol 'q' ditekan di terminal. Menghentikan program...")
                    break

            if is_headless:
                # Mode Headless: proses deteksi & absensi langsung dari frame tanpa render display
                for streamer in streamers:
                    raw_frame, connected, fps = streamer.get_latest_frame()
                    process_camera_frame(
                        raw_frame,
                        streamer.cam_name,
                        streamer.url,
                        connected,
                        fps,
                        known_faces_db,
                        face_detector,
                        recognizer,
                        det_size,
                        active_device,
                        render_display=False
                    )
                time.sleep(0.01)
            else:
                # Mode GUI Display
                tiles = []
                for streamer in streamers:
                    raw_frame, connected, fps = streamer.get_latest_frame()
                    tile = process_camera_frame(
                        raw_frame,
                        streamer.cam_name,
                        streamer.url,
                        connected,
                        fps,
                        known_faces_db,
                        face_detector,
                        recognizer,
                        det_size,
                        active_device,
                        render_display=True
                    )
                    tiles.append(tile)

                grid_frame = create_grid_display(tiles)
                cv2.imshow(window_name, grid_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nTombol 'q' ditekan di jendela GUI. Menghentikan program...")
                    break

    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna (Ctrl+C).")

    finally:
        print("\nMenutup koneksi kamera dan membersihkan resource...")
        for streamer in streamers:
            streamer.stop()
        if not is_headless:
            cv2.destroyAllWindows()
        print("Selesai.")


if __name__ == '__main__':
    main()
