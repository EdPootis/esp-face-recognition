import os
import sys
import glob
import time
import threading
import urllib.parse
from datetime import datetime
import cv2
import urllib.request
import urllib.error
import numpy as np
from PIL import Image, ImageOps

if sys.platform == 'win32':
    import msvcrt

# ==============================================================================
# KONFIGURASI KAMERA & MODEL
# ==============================================================================

# Daftar IP Address ESP32-CAM (Bisa ditambahkan sesuai kebutuhan)
CAMERAS = {
    'ESP32-CAM 1': 'http://172.16.108.130/capture',
    'ESP32-CAM 2': 'http://172.16.108.31/capture',
    'ESP32-S3': 'http://172.16.108.60/capture',
}

# Folder berisi foto referensi wajah
KNOWN_FACES_DIR = 'known_faces'

# Threshold kemiripan SFace (>= nilai ini dianggap orang yang sama)
RECOGNITION_THRESHOLD = 0.363

# Resolusi standar tiap tile kamera pada jendela gabungan (Grid View)
TILE_WIDTH = 480
TILE_HEIGHT = 360

# Model Paths
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

# Inisialisasi Model YuNet & SFace
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


# ==============================================================================
# CLASS MULTITHREADING STREAMER
# ==============================================================================
class CameraStreamer(threading.Thread):
    """
    Thread independen untuk tiap ESP32-CAM.
    Mengambil frame terbaru secara non-blocking agar delay pada satu kamera
    tidak mempengaruhi performa kamera lainnya.
    """
    def __init__(self, cam_name, url, timeout=2.5):
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

    def run(self):
        while self.running:
            try:
                req = urllib.request.Request(self.url, headers={'User-Agent': 'ESP32CAM-Client'})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw_data = resp.read()
                    img_np = np.array(bytearray(raw_data), dtype=np.uint8)
                    decoded = cv2.imdecode(img_np, -1)

                    if decoded is not None:
                        with self.lock:
                            self.frame = decoded
                            self.connected = True
                        self._frame_count += 1
                        
                        # Hitung FPS real-time per stream
                        now = time.time()
                        if now - self._fps_timer >= 1.0:
                            self.fps = self._frame_count / (now - self._fps_timer)
                            self._frame_count = 0
                            self._fps_timer = now
                    else:
                        with self.lock:
                            self.connected = False

                # Jeda mikro agar tidak membebani core CPU secara berlebihan
                time.sleep(0.005)

            except Exception:
                with self.lock:
                    self.connected = False
                # Jeda sebelum mencoba reconnect
                time.sleep(0.5)

    def get_latest_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.connected, self.fps
            return None, self.connected, self.fps

    def stop(self):
        self.running = False


# ==============================================================================
# FUNGSI EMBEDDING & PENGENALAN WAJAH
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

    print("--- Memulai Enrollment Database Wajah ---")
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
            except Exception as e:
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

        print(f"  [OK] '{person_name}': {count} embedding ter-enroll.")

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


# Dictionary untuk tracking waktu & nama terakhir dikirim per kamera
_cam_last_notif = {}


def send_face_to_esp32_async(cam_url, name, score, cam_name):
    """Kirim hasil deteksi (waktu, nama, kamera, skor) ke ESP32-CAM secara async."""
    def _worker():
        try:
            # cam_url misal 'http://172.16.108.31/capture' -> base_url 'http://172.16.108.31'
            base_url = cam_url.split('/capture')[0].rstrip('/')
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


# ==============================================================================
# PEMROSESAN & TAMPILAN GRID
# ==============================================================================
def process_camera_frame(raw_frame, cam_name, cam_url, connected, fps, known_faces_db):
    """
    Menjalankan face detection & recognition pada frame,
    lalu menambahkan overlay nama kamera, status koneksi, dan bounding box.
    """
    if not connected or raw_frame is None:
        # Tampilan placeholder jika kamera offline / belum tersambung
        tile = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
        tile[:] = (30, 30, 30)  # Warna abu-abu gelap
        cv2.putText(tile, cam_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(tile, "[ OFFLINE / RECONNECTING ]", (20, TILE_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return tile

    # Clone dan resize frame ke ukuran tile tampilan
    display_frame = cv2.resize(raw_frame, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)

    # Deteksi wajah pada frame asli
    h, w, _ = raw_frame.shape
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(raw_frame)

    scale_x = TILE_WIDTH / w
    scale_y = TILE_HEIGHT / h

    if faces is not None and len(faces) > 0:
        global _cam_last_notif
        for face in faces:
            fx, fy, fw, fh = map(int, face[:4])
            fx, fy = max(0, fx), max(0, fy)
            fw, fh = min(fw, w - fx), min(fh, h - fy)

            # Ekstrak embedding dan kenali wajah
            embedding = get_embedding(raw_frame, face)
            name, score = recognize(embedding, known_faces_db)

            # Kirim notifikasi ke Serial Monitor ESP32-CAM (cooldown 1.5 detik per kamera)
            now = time.time()
            last_name, last_time = _cam_last_notif.get(cam_name, ("", 0))
            if (now - last_time > 1.5) or (name != last_name):
                send_face_to_esp32_async(cam_url, name, score, cam_name)
                _cam_last_notif[cam_name] = (name, now)

            # Hitung koordinat skala untuk visualisasi pada tile
            dx = int(fx * scale_x)
            dy = int(fy * scale_y)
            dw = int(fw * scale_x)
            dh = int(fh * scale_y)

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_frame, (dx, dy), (dx + dw, dy + dh), color, 2)

            label = f"{name} ({score:.2f})"
            cv2.putText(display_frame, label, (dx, max(15, dy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Header status kamera (Nama + FPS + Indikator Hijau)
    cv2.rectangle(display_frame, (0, 0), (TILE_WIDTH, 28), (0, 0, 0), -1)
    cv2.circle(display_frame, (12, 14), 5, (0, 255, 0), -1)
    status_text = f"{cam_name} | {fps:.1f} FPS"
    cv2.putText(display_frame, status_text, (25, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return display_frame


def create_grid_display(tiles):
    """
    Menggabungkan daftar frame tile menjadi 1 jendela grid otomatis (1x2, 2x2, dst.)
    """
    n = len(tiles)
    if n == 0:
        return np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
    if n == 1:
        return tiles[0]
    if n == 2:
        return np.hstack((tiles[0], tiles[1]))

    # Jika 3 atau 4 kamera -> susun dalam grid 2x2
    cols = 2
    rows = (n + cols - 1) // cols

    # Jika ganjil, tambahkan tile kosong hitam sebagai penutup
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


# ==============================================================================
# MAIN LOOP
# ==============================================================================
def main():
    print("=" * 60)
    print("ESP32-CAM Multi-Stream Face Recognition System")
    print("=" * 60)

    # 1. Pendaftaran Database Wajah
    known_faces_db = enroll_known_faces()
    total_people = len(set(n for n, _ in known_faces_db))
    print(f"Total database: {len(known_faces_db)} foto dari {total_people} profil terdaftar.\n")

    # 2. Inisialisasi & Start Thread tiap Kamera
    streamers = []
    print("Memulai background thread untuk setiap kamera:")
    for cam_name, url in CAMERAS.items():
        print(f" -> Menghubungkan ke [{cam_name}]: {url}")
        streamer = CameraStreamer(cam_name, url)
        streamer.start()
        streamers.append(streamer)

    print("\nSemua kamera telah diaktifkan.")
    print("Tekan 'q' pada jendela GUI atau di terminal untuk keluar.\n")

    window_name = "Multi-Camera ESP32-CAM Face Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            # Pengecekan tombol 'q' dari terminal
            if sys.platform == 'win32' and msvcrt.kbhit():
                if msvcrt.getch().lower() == b'q':
                    print("\nTombol 'q' ditekan di terminal. Menghentikan program...")
                    break

            # 3. Ambil frame terbaru dari seluruh kamera & proses
            tiles = []
            for streamer in streamers:
                raw_frame, connected, fps = streamer.get_latest_frame()
                tile = process_camera_frame(raw_frame, streamer.cam_name, streamer.url, connected, fps, known_faces_db)
                tiles.append(tile)

            # 4. Gabungkan ke dalam 1 tampilan Grid
            grid_frame = create_grid_display(tiles)
            cv2.imshow(window_name, grid_frame)

            # Pengecekan tombol 'q' dari jendela GUI OpenCV
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
        cv2.destroyAllWindows()
        print("Selesai.")


if __name__ == '__main__':
    main()
