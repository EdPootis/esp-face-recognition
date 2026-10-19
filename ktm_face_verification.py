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
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageOps

if sys.platform == 'win32':
    import msvcrt

# ==============================================================================
# KONFIGURASI KAMERA & SISTEM VERIFIKASI 2FA (KTM + FACE RECOGNITION)
# ==============================================================================

# Daftar IP Address ESP32-CAM
CAMERA_HOSTS = {
    'ESP32-S3': '10.7.28.33',
    'ESP32-CAM 1': '10.244.226.130',
    'ESP32-CAM 2': '10.244.226.31',
}

def get_camera_urls(mode='stream'):
    """
    Menghasilkan dictionary URL kamera berdasarkan mode yang dipilih:
    - 'stream'  : http://<IP>:81/stream (MJPEG Continuous Stream via Port 81)
    - 'capture' : http://<IP>/capture (Snapshot Polling Keep-Alive via Port 80)
    """
    cameras = {}
    for name, host in CAMERA_HOSTS.items():
        if mode == 'stream':
            cameras[name] = f"http://{host}:81/stream"
        else:
            cameras[name] = f"http://{host}/capture"
    return cameras

# Default CAMERAS dictionary
CAMERAS = get_camera_urls('stream')

# Database Folder & File Mapping
KNOWN_FACES_DIR = 'known_faces'
STUDENTS_DB_FILE = 'students_database.json'

# Threshold Kemiripan SFace (>= nilai ini dianggap orang yang sama)
RECOGNITION_THRESHOLD = 0.363

# Durasi Jendela Waktu Verifikasi Wajah setelah Tap Kartu (Detik)
VERIFICATION_TIMEOUT = 8.0

# Durasi Tampilan Hasil Verifikasi (Sukses / Gagal / Timeout) sebelum GUI kembali ke Standby (Detik)
RESULT_DISPLAY_DURATION = 3.0

# Port HTTP Webhook Server (Untuk menerima notifikasi tap kartu dari ESP32 via WiFi)
TAP_SERVER_PORT = 5050

# Konfigurasi Dashboard Django (Log Absensi)
DASHBOARD_URL = 'http://10.7.28.21:8000/api/log/'
ENABLE_DASHBOARD = True

# Resolusi Tampilan Tile Kamera
TILE_WIDTH = 640
TILE_HEIGHT = 480

# Model AI YuNet & SFace
detect_model_path = 'face_detection_yunet_2023mar.onnx'
detect_model_url = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'

recog_model_path = 'face_recognition_sface_2021dec.onnx'
recog_model_url = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'

# Salinan SFace untuk ONNX Runtime (--rec-backend): bobot sama, tetapi initializer dihapus
# dari daftar graph input agar ONNX Runtime bisa constant folding. Output identik dengan
# OpenCV (selisih < 1e-5). Jika file tidak ada, file asli tetap dipakai.
recog_model_ort_path = 'face_recognition_sface_2021dec_ort.onnx'

# Unduh model jika belum ada
for path, dl_url in [(detect_model_path, detect_model_url), (recog_model_path, recog_model_url)]:
    if not os.path.exists(path):
        print(f"Mengunduh model {path}...")
        urllib.request.urlretrieve(dl_url, path)
        print(f"Model {path} berhasil diunduh.")

face_detector = cv2.FaceDetectorYN_create(
    model=detect_model_path,
    config='',
    input_size=(800, 600),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

enroll_detector = cv2.FaceDetectorYN_create(
    model=detect_model_path,
    config='',
    input_size=(800, 600),
    score_threshold=0.3,
    nms_threshold=0.3,
    top_k=5000
)

recognizer = cv2.FaceRecognizerSF_create(model=recog_model_path, config='')


class SFaceORTRecognizer:
    """
    SFace dengan inferensi via ONNX Runtime. Antarmuka sama dengan cv2.FaceRecognizerSF:
    alignCrop() dan match() tetap dari OpenCV, hanya feature() (inferensi model, bagian
    terberat) yang dijalankan ONNX Runtime.
    Hasil benchmark di Jetson Nano: recognition 185 ms (OpenCV DNN) -> 37 ms (ONNX Runtime
    GPU) dengan embedding identik. Lihat benchmark/benchmark_01_yunet_sface.py --rec-backend ort.
    """

    def __init__(self, cv_recognizer, ort_module, providers):
        model_path = recog_model_ort_path if os.path.exists(recog_model_ort_path) else recog_model_path
        if model_path == recog_model_path:
            print(f"[WARN] {recog_model_ort_path} tidak ditemukan, memakai model asli "
                  f"(ONNX Runtime lebih lambat karena tidak bisa constant folding).")

        sess_options = ort_module.SessionOptions()
        sess_options.log_severity_level = 3  # sembunyikan warning per-initializer
        self.session = ort_module.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.provider = self.session.get_providers()[0]
        self.device = 'GPU' if self.provider == 'CUDAExecutionProvider' else 'CPU'
        self._cv = cv_recognizer

        # Warm-up (run pertama CUDA jauh lebih lambat karena inisialisasi kernel)
        for _ in range(3):
            self.feature(np.zeros((112, 112, 3), dtype=np.uint8))

    def alignCrop(self, img, face):
        return self._cv.alignCrop(img, face)

    def feature(self, aligned_img):
        # Preprocessing identik dengan FaceRecognizerSF::feature() di OpenCV
        blob = cv2.dnn.blobFromImage(aligned_img, 1.0, (112, 112), (0, 0, 0), swapRB=True, crop=False)
        return self.session.run(None, {self.input_name: blob})[0]

    def match(self, feature1, feature2, dis_type):
        return self._cv.match(feature1, feature2, dis_type)


def load_ort_recognizer(rec_backend, cv_recognizer):
    """
    Memuat SFace via ONNX Runtime sesuai --rec-backend. Mengembalikan None jika SFace
    tetap memakai OpenCV DNN.
    - 'opencv': selalu OpenCV DNN.
    - 'auto'  : ONNX Runtime jika terpasang (GPU jika ada, selain itu CPU), selain itu
                OpenCV DNN. Di CPU laptop pun feature() ±2x lebih cepat (±7 ms vs ±15 ms);
                di Raspberry Pi belum diuji.
    - 'ort'   : sama dengan 'auto', tetapi memberi peringatan jika onnxruntime tidak ada.
    """
    if rec_backend == 'opencv':
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        if rec_backend == 'ort':
            print("[WARN] onnxruntime belum terpasang, SFace memakai OpenCV DNN.")
        return None

    has_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
    providers = (['CUDAExecutionProvider'] if has_cuda else []) + ['CPUExecutionProvider']
    try:
        return SFaceORTRecognizer(cv_recognizer, ort, providers)
    except Exception as e:
        print(f"[WARN] ONNX Runtime gagal dimuat ({e}), SFace memakai OpenCV DNN.")
        return None


# ==============================================================================
# MANAJEMEN DATABASE MAHASISWA & PEMETAAN UID KTM
# ==============================================================================
class StudentDatabase:
    """
    Mengelola pemetaan antara UID Kartu KTM dengan Biodata Mahasiswa & Folder Foto Wajah.
    Format disimpan dalam students_database.json.
    """
    def __init__(self, json_path=STUDENTS_DB_FILE):
        self.json_path = json_path
        self.students = {}
        self.load()

    def load(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.students = data.get('students', {})
            except Exception as e:
                print(f"[WARN] Gagal membaca {self.json_path}: {e}")
                self.students = {}
        else:
            self.save_default()

    def save_default(self):
        default_data = {
            "_comment": "Database Pemetaan UID Kartu KTM ke Profil Mahasiswa untuk Verifikasi 2FA",
            "students": {
                "DEA35B89": {"nim": "2306208363", "name": "Edmond", "folder": "edmond", "major": "TEsting "},
                "04A12B9C": {"nim": "13520002", "name": "Juan", "folder": "juan", "major": "HihiHAHA"},
                "A1B2C3D4": {"nim": "13520003", "name": "Topa", "folder": "topa", "major": "Sistem Informasi"}
            }
        }
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(default_data, f, indent=2)
        self.students = default_data["students"]

    def get_student_by_uid(self, uid):
        clean_uid = uid.replace(" ", "").upper()
        return self.students.get(clean_uid, None)

    def get_uid_by_folder(self, folder_name):
        for uid, info in self.students.items():
            if info.get('folder', '').lower() == folder_name.lower():
                return uid
        return None


student_db = StudentDatabase()


# ==============================================================================
# STATE MACHINE VERIFIKASI 2FA
# ==============================================================================
class VerificationState:
    IDLE = "IDLE"                            # Standby: Menunggu Mahasiswa Tap Kartu
    VERIFYING = "VERIFYING"                  # Kartu di-tap: Memeriksa Wajah di Depan Kamera
    SUCCESS = "SUCCESS"                      # Wajah Cocok dengan Pemilik Kartu (Verified!)
    REJECTED_MISMATCH = "REJECTED_MISMATCH"  # Wajah TIDAK Cocok (Percobaan Titip Absen / Joki!)
    REJECTED_UNKNOWN = "REJECTED_UNKNOWN"    # Kartu Tidak Terdaftar di Database
    TIMEOUT_NO_FACE = "TIMEOUT_NO_FACE"      # Waktu Habis tanpa Wajah Terdeteksi


class VerificationManager:
    """
    Mengontrol alur logika 2-Factor Authentication:
    1. Mahasiswa Tap Kartu -> Menerima UID.
    2. Mencari data pemilik kartu di students_database.json.
    3. Mengaktifkan jendela waktu verifikasi (8 detik).
    4. Mencocokkan wajah live dengan profil pemilik kartu.
    """
    def __init__(self):
        self.state = VerificationState.IDLE
        self.current_uid = None
        self.student_info = None
        self.state_start_time = 0.0
        self.last_result_time = 0.0
        self.result_message = ""
        self.matched_score = 0.0
        self.detected_person = None
        self.cam_name = "ESP32-S3"
        self.lock = threading.Lock()

    def trigger_card_tap(self, uid, cam_name="ESP32-S3"):
        """Dipanggil saat kartu di-tap pada Card Reader."""
        with self.lock:
            clean_uid = uid.replace(" ", "").upper()
            student = student_db.get_student_by_uid(clean_uid)

            self.cam_name = cam_name
            self.current_uid = clean_uid
            self.state_start_time = time.time()
            self.matched_score = 0.0
            self.detected_person = None

            if student:
                self.student_info = student
                self.state = VerificationState.VERIFYING
                self.result_message = f"Halo {student['name']} ({student['nim']})! Silakan lihat kamera..."
                print(f"\n[TAP KTM] Terbaca UID: {clean_uid} -> {student['name']} ({student['nim']})")
                print(f" -> Memulai verifikasi wajah... (Timeout: {VERIFICATION_TIMEOUT}s)")
                
                # Kirim notifikasi ke LCD ESP32
                send_face_to_esp32_async(CAMERAS.get(cam_name, ''), student['name'], 0.0, cam_name)
            else:
                # State diubah lebih dulu: GUI membaca student_info saat state == VERIFYING
                self.state = VerificationState.REJECTED_UNKNOWN
                self.student_info = None
                self.last_result_time = time.time()
                self.result_message = f"KARTU TIDAK TERDAFTAR! (UID: {clean_uid})"
                print(f"\n[TAP KTM REJECTED] UID {clean_uid} tidak terdaftar di database!")
                send_face_to_esp32_async(CAMERAS.get(cam_name, ''), "KARTU TDK DIKENAL", 0.0, cam_name)

    def update(self):
        """
        Dipanggil setiap iterasi main loop, terlepas dari ada/tidaknya frame atau wajah:
        1. Mengakhiri VERIFYING saat jendela waktu verifikasi habis.
        2. Mengembalikan GUI ke IDLE setelah hasil ditampilkan RESULT_DISPLAY_DURATION detik.
        """
        with self.lock:
            now = time.time()

            if self.state == VerificationState.VERIFYING:
                if now - self.state_start_time > VERIFICATION_TIMEOUT:
                    self.state = VerificationState.TIMEOUT_NO_FACE
                    self.last_result_time = now
                    self.result_message = "TIMEOUT! Wajah tidak terdeteksi / tidak cocok."
                    print(f"[VERIFIKASI GAGAL] Timeout {VERIFICATION_TIMEOUT}s habis untuk {self.student_info['name']}.")
                    send_face_to_esp32_async(CAMERAS.get(self.cam_name, ''), "TIMEOUT!", 0.0, self.cam_name)

            elif self.state != VerificationState.IDLE:
                if now - self.last_result_time > RESULT_DISPLAY_DURATION:
                    self.state = VerificationState.IDLE
                    self.current_uid = None
                    self.student_info = None
                    self.result_message = ""
                    self.matched_score = 0.0
                    self.detected_person = None

    def process_face_recognition(self, recognized_name, score, raw_frame, cam_name):
        """Memvalidasi wajah yang terdeteksi saat dalam status VERIFYING."""
        with self.lock:
            now = time.time()

            # Timeout & reset ke IDLE ditangani update(); di sini hanya pencocokan wajah
            if self.state == VerificationState.VERIFYING:
                if now - self.state_start_time > VERIFICATION_TIMEOUT:
                    return

                # Jika belum ada wajah terdeteksi pada frame ini, tunggu frame berikutnya
                if recognized_name is None:
                    return

                expected_folder = self.student_info.get('folder', '').lower()
                expected_name = self.student_info.get('name', '')

                # KASUS 1: WAJAH COCOK (VERIFIED!)
                if recognized_name.lower() == expected_folder:
                    self.state = VerificationState.SUCCESS
                    self.last_result_time = now
                    self.matched_score = score
                    self.detected_person = expected_name
                    self.result_message = f"VERIFIKASI SUKSES! {expected_name} (Skor: {score:.2f})"
                    print(f"\n[VERIFIKASI SUKSES] ✓ Wajah {expected_name} terverifikasi! (Skor: {score:.2f})")

                    # Kirim data kehadiran ke Dashboard Django (Format LogPresentronik)
                    send_log_to_dashboard(
                        cam_name=cam_name,
                        person_name=expected_name,
                        similarity=score,
                        recognized=True,
                        face_image=raw_frame,
                        nim=self.student_info.get('nim', '-'),
                        uid=self.current_uid,
                        kd_org=self.student_info.get('kd_org', self.student_info.get('major', '01.00.12.01')),
                        status='HADIR'
                    )

                    # Update tampilan LCD ESP32
                    send_face_to_esp32_async(CAMERAS.get(cam_name, ''), expected_name, score, cam_name)

                # KASUS 2: WAJAH MILIK ORANG LAIN (PERCOBAAN TITIP ABSEN / JOKI!)
                elif recognized_name != "Unknown" and recognized_name.lower() != expected_folder:
                    self.state = VerificationState.REJECTED_MISMATCH
                    self.last_result_time = now
                    self.matched_score = score
                    self.detected_person = recognized_name
                    self.result_message = f"DITOLAK! Wajah terdeteksi sebagai '{recognized_name}', bukan '{expected_name}'!"
                    print(f"\n[🚨 PERINGATAN KECURANGAN] Kartu milik {expected_name}, tapi wajah yang berdiri di depan kamera adalah {recognized_name}!")
                    
                    send_face_to_esp32_async(CAMERAS.get(cam_name, ''), "WAJAH TDK COCOK!", score, cam_name)


verif_manager = VerificationManager()


# ==============================================================================
# HTTP SERVER UNTUK MENERIMA WEBHOOK TAP KARTU DARI ESP32
# ==============================================================================
class TapWebhookHandler(BaseHTTPRequestHandler):
    """
    Server HTTP lokal agar ESP32 dapat mengirim data tap kartu secara langsung:
    Contoh: GET http://[IP_KOMPUTER]:5050/tap?uid=DEA35B89&cam=ESP32-S3
    """
    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path in ('/tap', '/api/tap'):
            query = urllib.parse.parse_qs(parsed_url.query)
            uid = query.get('uid', [None])[0]
            cam = query.get('cam', ['ESP32-S3'])[0]

            if uid:
                verif_manager.trigger_card_tap(uid, cam)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                resp = json.dumps({"status": "ok", "uid": uid, "message": "Tap received"}).encode('utf-8')
                self.wfile.write(resp)
                return

        self.send_response(400)
        self.end_headers()

    def do_POST(self):
        if self.path in ('/tap', '/api/tap'):
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                uid = data.get('uid')
                cam = data.get('cam', 'ESP32-S3')
                if uid:
                    verif_manager.trigger_card_tap(uid, cam)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                    return
            except Exception:
                pass
        self.send_response(400)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Matikan logging request HTTP agar console tetap bersih


def start_tap_server(port=TAP_SERVER_PORT):
    try:
        server = HTTPServer(('0.0.0.0', port), TapWebhookHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"✓ Tap Webhook Server aktif di port {port} (Siap menerima tap dari ESP32)")
    except Exception as e:
        print(f"[WARN] Gagal memulai Tap Webhook Server di port {port}: {e}")


# ==============================================================================
# CLASS MULTITHREADING STREAMER (Sama dengan multi_esp32cam.py)
# ==============================================================================
# Ada sejak Python 3.5; read1() mengembalikan data yang sudah tiba tanpa menunggu buffer penuh
_HTTP_HAS_READ1 = 'read1' in http.client.HTTPResponse.__dict__

# Batas buffer stream tanpa marker EOI (data rusak / bukan JPEG) sebelum dibuang
MAX_STREAM_BUFFER = 2 * 1024 * 1024


class CameraStreamer(threading.Thread):
    """
    Thread pembaca kamera. Thread ini hanya mengambil JPEG mentah terbaru dari socket
    agar socket selalu terkuras (ESP32 tidak tertahan saat mengirim). Decode JPEG
    dilakukan di get_latest_frame(), hanya untuk frame yang benar-benar dipakai.
    """
    def __init__(self, cam_name, url, timeout=4.0):
        super().__init__(daemon=True)
        self.cam_name = cam_name
        self.url = url
        self.timeout = timeout
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()

        # JPEG mentah terbaru dari thread pembaca, di-decode secara lazy
        self._jpeg = None
        self._jpeg_seq = 0
        self._decoded_seq = 0
        self.decode_errors = 0

        # Diagnostik putus-sambung stream
        self.last_error = None
        self.disconnect_count = 0
        self._disconnected_since = None

        parsed = urllib.parse.urlparse(self.url)
        self.host = parsed.netloc
        self.path = parsed.path if parsed.path else '/capture'
        if parsed.query:
            self.path += '?' + parsed.query
        self.conn = None
        self.mode = 'stream' if '/stream' in self.path else 'capture'

    def _get_connection(self):
        if self.conn is None:
            self.conn = http.client.HTTPConnection(self.host, timeout=self.timeout)
        return self.conn

    def _store_jpeg(self, jpg_data):
        with self.lock:
            was_connected = self.connected
            self._jpeg = jpg_data
            self._jpeg_seq += 1
            self.connected = True

        if not was_connected:
            if self._disconnected_since is not None:
                down_for = time.time() - self._disconnected_since
                print(f"[STREAM {self.cam_name}] Tersambung kembali setelah {down_for:.1f} detik "
                      f"(putus ke-{self.disconnect_count})")
            else:
                print(f"[STREAM {self.cam_name}] Tersambung ke {self.url}")
            self._disconnected_since = None
            self.last_error = None
            self._frame_count = 0
            self._fps_timer = time.time()

        self._frame_count += 1
        now = time.time()
        if now - self._fps_timer >= 1.0:
            self.fps = self._frame_count / (now - self._fps_timer)
            self._frame_count = 0
            self._fps_timer = now

    def _mark_disconnected(self, reason):
        with self.lock:
            was_connected = self.connected
            self.connected = False

        if was_connected:
            self.disconnect_count += 1
            self._disconnected_since = time.time()
            print(f"[STREAM {self.cam_name}] Stream putus: {reason} -> mencoba reconnect...")
        elif reason != self.last_error:
            # Saat reconnect terus gagal, cetak hanya jika penyebabnya berubah agar log tidak banjir
            if self._disconnected_since is None:
                self._disconnected_since = time.time()
            print(f"[STREAM {self.cam_name}] Belum tersambung: {reason}")
        self.last_error = reason

    def _run_stream(self):
        conn = self._get_connection()
        conn.request('GET', self.path, headers={'User-Agent': 'ESP32CAM-Client'})
        resp = conn.getresponse()

        if resp.status != 200:
            resp.read()
            self._mark_disconnected(f"HTTP {resp.status} {resp.reason}")
            time.sleep(0.5)
            return

        if _HTTP_HAS_READ1:
            read_chunk = lambda: resp.read1(65536)
        else:
            read_chunk = lambda: resp.read(4096)

        buf = bytearray()
        while self.running:
            chunk = read_chunk()
            if not chunk:
                raise ConnectionError("koneksi ditutup oleh kamera")
            buf += chunk

            # Ambil JPEG lengkap terbaru; frame lama yang menumpuk di buffer dilewati
            end = buf.rfind(b'\xff\xd9')
            if end == -1:
                if len(buf) > MAX_STREAM_BUFFER:
                    del buf[:]
                continue
            start = buf.rfind(b'\xff\xd8', 0, end)
            if start != -1:
                self._store_jpeg(bytes(buf[start:end + 2]))
            del buf[:end + 2]

    def _run_capture(self):
        conn = self._get_connection()
        conn.request('GET', self.path, headers={
            'User-Agent': 'ESP32CAM-Client',
            'Connection': 'keep-alive'
        })
        resp = conn.getresponse()

        if resp.status == 200:
            self._store_jpeg(resp.read())
        else:
            resp.read()
            self._mark_disconnected(f"HTTP {resp.status} {resp.reason}")
            time.sleep(0.5)

        time.sleep(0.005)

    def run(self):
        while self.running:
            try:
                if self.mode == 'stream':
                    self._run_stream()
                else:
                    self._run_capture()
            except Exception as e:
                if not self.running:
                    break
                self._mark_disconnected(f"{type(e).__name__}: {e}")
                if self.conn:
                    try:
                        self.conn.close()
                    except Exception:
                        pass
                    self.conn = None
                time.sleep(0.5)

    def get_latest_frame(self):
        with self.lock:
            jpg = self._jpeg
            seq = self._jpeg_seq
            connected = self.connected
            fps = self.fps

        # Decode hanya jika ada JPEG baru sejak pemanggilan terakhir
        if jpg is not None and seq != self._decoded_seq:
            self._decoded_seq = seq
            decoded = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is not None:
                self.frame = decoded
            else:
                # JPEG rusak satu frame bukan berarti stream putus; pakai frame sebelumnya
                self.decode_errors += 1

        if self.frame is not None:
            return self.frame.copy(), connected, fps
        return None, connected, fps

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


def get_embedding(image, face_row):
    aligned_face = recognizer.alignCrop(image, face_row)
    return recognizer.feature(aligned_face)


def enroll_known_faces():
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
            embedding = get_embedding(img, best_face)
            database.append((person_name, embedding))
            count += 1

        uid_mapped = student_db.get_uid_by_folder(person_name)
        uid_tag = f" [UID: {uid_mapped}]" if uid_mapped else " [Belum ada UID di JSON]"
        print(f"  [OK] '{person_name}'{uid_tag}: {count} embedding ter-enroll.")

    return database


def recognize(embedding, database):
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


def send_face_to_esp32_async(cam_url, name, score, cam_name):
    """Mengirim hasil deteksi nama/waktu ke ESP32 agar LCD 16x2 terupdate."""
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


def send_log_to_dashboard(cam_name, person_name, similarity, recognized, face_image=None, nim="-", uid="-", kd_org="01.00.12.01", status="HADIR"):
    """
    Mengirim log kehadiran ke Django Dashboard API sesuai skema model LogPresentronik.
    """
    if not ENABLE_DASHBOARD:
        return

    def _worker():
        try:
            foto_b64 = None
            if face_image is not None:
                success, encoded_img = cv2.imencode('.jpg', face_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    raw_b64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
                    foto_b64 = f"data:image/jpeg;base64,{raw_b64}"

            now_dt = datetime.now()
            mac_addr = CAMERA_HOSTS.get(cam_name, '00:00:00:00:00:00')

            payload = {
                # Format Standar Model LogPresentronik
                'no_identitas': nim if nim else '-',
                'nama': person_name,
                'rfid_tag': uid if uid else '-',
                'mac_address': mac_addr,
                'kd_org': kd_org if kd_org else '01.00.12.01',
                'tipe_perangkat': 'ESP32-S3 Face Recognition',
                'ts_presensi_device': now_dt.strftime('%Y-%m-%d %H:%M:%S'),

                # Atribut Tambahan untuk Monitoring & Audit Face Recognition
                'kamera': cam_name,
                'skor': round(float(similarity), 4),
                'status': status,
                'foto_b64': foto_b64,

                # Alias Kompatibilitas
                'nim': nim if nim else '-',
                'person_name': person_name,
                'similarity': round(float(similarity), 4),
                'recognized': recognized,
                'timestamp': now_dt.isoformat(),
                'image_base64': foto_b64
            }

            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                DASHBOARD_URL,
                data=data,
                headers={'Content-Type': 'application/json', 'User-Agent': 'FaceRecClient'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                status_code = resp.getcode()
                print(f"[DASHBOARD] Log kehadiran terkirim ({status_code}) -> {person_name} ({nim}) | UID: {uid} | Skor: {similarity:.2f}")
        except Exception as e:
            print(f"[DASHBOARD ERROR] Gagal mengirim log ({cam_name} / {person_name}): {e}")

    threading.Thread(target=_worker, daemon=True).start()


# ==============================================================================
# RENDERING GUI & PROCESS FRAME
# ==============================================================================
def process_camera_frame(raw_frame, cam_name, connected, fps, known_faces_db):
    if not connected or raw_frame is None:
        tile = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
        tile[:] = (30, 30, 30)
        cv2.putText(tile, cam_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(tile, "[ OFFLINE / RECONNECTING ]", (20, TILE_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return tile

    display_frame = cv2.resize(raw_frame, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)
    h, w, _ = raw_frame.shape
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(raw_frame)

    scale_x = TILE_WIDTH / w
    scale_y = TILE_HEIGHT / h

    recognized_person_in_frame = None
    recognized_score_in_frame = 0.0

    if faces is not None:
        for face in faces:
            box = face[0:4].astype(int)
            x, y, bw, bh = box

            x = max(0, x)
            y = max(0, y)
            bw = min(w - x, bw)
            bh = min(h - y, bh)

            if bw <= 0 or bh <= 0:
                continue

            embedding = get_embedding(raw_frame, face)
            name, score = recognize(embedding, known_faces_db)

            recognized_person_in_frame = name
            recognized_score_in_frame = score

            # Skalakan koordinat bounding box ke ukuran display
            dx = int(x * scale_x)
            dy = int(y * scale_y)
            dbw = int(bw * scale_x)
            dbh = int(bh * scale_y)

            # Warna kotak berdasarkan state verifikasi
            state = verif_manager.state
            if state == VerificationState.VERIFYING:
                expected = verif_manager.student_info.get('folder', '').lower()
                if name.lower() == expected:
                    color = (0, 255, 0)      # Hijau (Cocok!)
                    status_txt = f"{verif_manager.student_info['name']} ({score:.2f})"
                else:
                    color = (0, 165, 255)    # Orange (Memeriksa / Mismatch)
                    status_txt = f"Wajah: {name} ({score:.2f})"
            elif state == VerificationState.SUCCESS:
                color = (0, 255, 0)          # Hijau
                status_txt = f"VERIFIED: {name}"
            elif state == VerificationState.REJECTED_MISMATCH:
                color = (0, 0, 255)          # Merah
                status_txt = "DITOLAK: Wajah Beda"
            else:
                color = (255, 200, 0)        # Kuning Standby
                status_txt = "Silakan Tap KTM"

            cv2.rectangle(display_frame, (dx, dy), (dx + dbw, dy + dbh), color, 2)
            cv2.putText(display_frame, status_txt, (dx, max(20, dy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Masukkan hasil pembacaan wajah ke State Machine Verifikasi 2FA
    if verif_manager.state == VerificationState.VERIFYING and recognized_person_in_frame is not None:
        verif_manager.process_face_recognition(
            recognized_person_in_frame,
            recognized_score_in_frame,
            raw_frame,
            cam_name
        )

    # Overlay Info Kamera & FPS di pojok atas
    cv2.putText(display_frame, f"{cam_name} | {fps:.1f} FPS", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return display_frame


def render_verification_banner(frame_width):
    """Membuat Banner Status Verifikasi di bagian bawah jendela."""
    banner_height = 110
    banner = np.zeros((banner_height, frame_width, 3), dtype=np.uint8)

    state = verif_manager.state
    now = time.time()

    if state == VerificationState.IDLE:
        banner[:] = (45, 30, 20)  # Biru Gelap
        cv2.putText(banner, "STATUS: STANDBY", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(banner, ">> SILAKAN TEMPELKAN KARTU TANDA MAHASISWA (KTM) PADA CARD READER <<",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)

    elif state == VerificationState.VERIFYING:
        banner[:] = (20, 45, 60)  # Kuning / Oranye Gelap
        time_left = max(0.0, VERIFICATION_TIMEOUT - (now - verif_manager.state_start_time))
        student = verif_manager.student_info

        cv2.putText(banner, f"STATUS: MEMERIKSA WAJAH (Sisa Waktu: {time_left:.1f}s)", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(banner, f"Mahasiswa: {student['name']} | NIM: {student['nim']} | Jurusan: {student.get('major', '-')}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(banner, "Silakan pandang lurus ke lensa kamera...", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 230, 255), 1)

        # Progress bar countdown
        bar_w = int((time_left / VERIFICATION_TIMEOUT) * (frame_width - 40))
        cv2.rectangle(banner, (20, 102), (20 + bar_w, 106), (0, 255, 255), -1)

    elif state == VerificationState.SUCCESS:
        banner[:] = (20, 60, 20)  # Hijau
        cv2.putText(banner, "✓ VERIFIKASI BERHASIL! (AKSES DITERIMA)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        cv2.putText(banner, verif_manager.result_message, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    elif state == VerificationState.REJECTED_MISMATCH:
        banner[:] = (20, 20, 70)  # Merah
        cv2.putText(banner, "🚨 KECURANGAN TERDETEKSI: WAJAH TIDAK COCOK! (AKSES DITOLAK)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(banner, verif_manager.result_message, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif state == VerificationState.REJECTED_UNKNOWN:
        banner[:] = (20, 20, 70)  # Merah
        cv2.putText(banner, "❌ KARTU TIDAK TERDAFTAR!", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
        cv2.putText(banner, verif_manager.result_message, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

    elif state == VerificationState.TIMEOUT_NO_FACE:
        banner[:] = (20, 40, 70)  # Oranye / Merah
        cv2.putText(banner, "⏱️ VERIFIKASI GAGAL: WAKTU HABIS (TIMEOUT)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(banner, "Tidak ada wajah yang cocok dalam 8 detik. Silakan tap kartu kembali.",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return banner


# ==============================================================================
# MAIN PROGRAM
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Sistem Verifikasi Absensi 2FA: KTM RFID/NFC + Face Recognition",
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
    parser.add_argument('--cam', type=str, default='ESP32-S3', help="Nama kamera utama yang digunakan (Default: ESP32-S3)")
    parser.add_argument('--port', type=int, default=TAP_SERVER_PORT, help="Port Webhook Tap Kartu (Default: 5050)")
    parser.add_argument(
        '--rec-backend', choices=['auto', 'opencv', 'ort'], default='auto',
        help="Runtime inferensi SFace:\n"
             "  'auto'   : ONNX Runtime jika terpasang, selain itu OpenCV DNN (Default)\n"
             "  'opencv' : OpenCV DNN\n"
             "  'ort'    : ONNX Runtime (GPU jika ada, selain itu CPU)"
    )
    return parser.parse_args()


def main():
    global CAMERAS, recognizer
    args = parse_args()
    raw_mode = args.mode_opt or args.mode_pos or 'stream'
    mode = 'stream' if raw_mode.lower() in ['s', 'stream'] else 'capture'
    CAMERAS = get_camera_urls(mode)

    # YuNet tetap di OpenCV DNN CPU; SFace dipindah ke ONNX Runtime jika tersedia.
    ort_recognizer = load_ort_recognizer(args.rec_backend, recognizer)
    if ort_recognizer is not None:
        recognizer = ort_recognizer
        rec_label = f"ONNX Runtime ({ort_recognizer.provider})"
    else:
        rec_label = "OpenCV DNN (CPU)"

    print("=" * 65)
    print("  SISTEM VERIFIKASI 2-FACTOR (KTM NFC/RFID + FACE RECOGNITION)")
    print(f"  Mode Endpoint: {mode.upper()} ({'/stream (Port 81)' if mode == 'stream' else '/capture (Port 80)'})")
    print(f"  Runtime AI   : YuNet OpenCV DNN (CPU) | SFace {rec_label}")
    print("=" * 65)

    # 1. Mulai Webhook Server Tap Kartu
    start_tap_server(args.port)

    # 2. Enroll database wajah
    known_faces_db = enroll_known_faces()
    print(f"Total Database: {len(known_faces_db)} embedding dari {len(student_db.students)} mahasiswa terdaftar.")

    # 3. Jalankan Camera Streamer
    streamers = {}
    for name, url in CAMERAS.items():
        print(f" -> Menghubungkan ke [{name}]: {url}")
        s = CameraStreamer(name, url, timeout=4.0)
        s.start()
        streamers[name] = s

    window_name = "2FA Attendance: KTM Tap + Face Recognition Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n✓ Semua sistem siap!")
    print("  • Tap kartu fisik pada modul PN532 / RC522")
    print("  • Atau ketik 't' pada GUI untuk simulasi tap kartu")
    print("  • Tekan 'q' pada GUI untuk keluar.\n")

    try:
        while True:
            # Timeout verifikasi & reset GUI ke standby (tidak bergantung pada frame kamera)
            verif_manager.update()

            # Render frame kamera
            tiles = []
            for cam_name, streamer in streamers.items():
                frame, connected, fps = streamer.get_latest_frame()
                tile = process_camera_frame(frame, cam_name, connected, fps, known_faces_db)
                tiles.append(tile)

            # Buat grid kamera
            if len(tiles) == 1:
                camera_grid = tiles[0]
            elif len(tiles) == 2:
                camera_grid = np.hstack(tiles)
            elif len(tiles) <= 4:
                top_row = np.hstack(tiles[:2])
                bottom_tiles = tiles[2:]
                while len(bottom_tiles) < 2:
                    blank = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
                    bottom_tiles.append(blank)
                bottom_row = np.hstack(bottom_tiles)
                camera_grid = np.vstack([top_row, bottom_row])
            else:
                camera_grid = tiles[0]

            # Render Status Banner di bagian bawah
            banner = render_verification_banner(camera_grid.shape[1])
            final_display = np.vstack([camera_grid, banner])

            cv2.imshow(window_name, final_display)

            # Handle Keyboard Input
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:
                print("Menghentikan program...")
                break
            elif key == ord('t'):
                # Simulasi Tap Kartu Manual (Untuk testing tanpa hardware)
                sample_uids = list(student_db.students.keys())
                if sample_uids:
                    simulated_uid = sample_uids[0]
                    print(f"\n[SIMULASI] Men-tap kartu contoh: UID '{simulated_uid}' ({student_db.students[simulated_uid]['name']})")
                    verif_manager.trigger_card_tap(simulated_uid, args.cam)

    finally:
        for streamer in streamers.values():
            streamer.stop()
        cv2.destroyAllWindows()
        print("Selesai.")


if __name__ == '__main__':
    main()
