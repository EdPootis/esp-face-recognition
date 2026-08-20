import os
import sys
import glob
import time
import threading
import json
import base64
import urllib.parse
from datetime import datetime
import cv2
import urllib.request
import urllib.error
import numpy as np
from PIL import Image, ImageOps

if sys.platform == 'win32':
    import msvcrt


# Ganti dengan IP Address ESP32-CAM Anda yang sesuai
#IP_ADDRESS = ['172.16.108.130', '172.16.108.31', '172.16.108.60']
IP_ADDRESS = '172.16.108.31'

# Nama/Label Kamera
CAMERA_NAME = f"ESP32-CAM ({IP_ADDRESS})"

url = f'http://{IP_ADDRESS}/capture'

# Folder berisi foto referensi, struktur: known_faces/<nama>/foto1.jpg, foto2.jpg, ...
KNOWN_FACES_DIR = 'known_faces'

# Threshold cosine similarity resmi dari OpenCV (>= nilai ini dianggap orang yang sama)
RECOGNITION_THRESHOLD = 0.363

# Konfigurasi Dashboard Django (Log Absensi)
DASHBOARD_URL = 'http://127.0.0.1:8000/api/receive-log/'
ENABLE_DASHBOARD = True
DASHBOARD_COOLDOWN_RECOGNIZED = 5.0  # Jeda pengiriman log wajah teridentifikasi (detik) per orang
DASHBOARD_COOLDOWN_UNKNOWN = 3.0     # Jeda pengiriman log wajah unknown (detik)

# Waktu & nama terakhir dikirim ke ESP32 (agar tidak spamming Serial Monitor)
_last_sent_name = ""
_last_sent_time = 0


def send_face_to_esp32_async(ip_address, name, score, cam_name=CAMERA_NAME):
    """Kirim hasil deteksi (waktu, nama, kamera, skor) ke ESP32-CAM secara async."""
    def _worker():
        try:
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            encoded_name = urllib.parse.quote(name)
            encoded_cam = urllib.parse.quote(cam_name)
            encoded_time = urllib.parse.quote(time_str)

            target_url = (f"http://{ip_address}/face?"
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


# Dictionary untuk tracking waktu log dikirim ke dashboard
_dashboard_last_sent = {}


def crop_face_roi(image, face_row, margin=0.2):
    """Crop area wajah dari image berdasarkan koordinat deteksi YuNet dengan margin tambahan."""
    h, w = image.shape[:2]
    fx, fy, fw, fh = map(int, face_row[:4])

    margin_w = int(fw * margin)
    margin_h = int(fh * margin)

    x1 = max(0, fx - margin_w)
    y1 = max(0, fy - margin_h)
    x2 = min(w, fx + fw + margin_w)
    y2 = min(h, fy + fh + margin_h)

    return image[y1:y2, x1:x2]


def encode_image_to_b64(crop_img, quality=85):
    """Encode OpenCV image/crop ke string Base64 format data URI JPEG."""
    if crop_img is None or crop_img.size == 0:
        return None
    success, buffer = cv2.imencode('.jpg', crop_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return None
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"


def send_log_to_dashboard_async(dashboard_url, cam_name, name, score, face_status, face_crop):
    """Kirim log absensi ke Dashboard Django (receive_log API) secara async."""
    def _worker():
        try:
            foto_b64 = encode_image_to_b64(face_crop)
            payload = {
                'nama': name,
                'kamera': cam_name,
                'skor': round(float(score), 4),
                'status': face_status,
                'foto_b64': foto_b64
            }
            data_bytes = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                dashboard_url,
                data=data_bytes,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'FaceRecClient'
                },
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                status_code = resp.getcode()
                print(f"[DASHBOARD] Log terkirim ({status_code}) -> Kamera: {cam_name} | Nama: {name} | Status: {face_status} | Skor: {score:.2f}")
        except Exception as e:
            print(f"[DASHBOARD ERROR] Gagal mengirim log ({cam_name} / {name}): {e}")

    threading.Thread(target=_worker, daemon=True).start()

# --- Path dan URL model Face Detection (YuNet) ---
detect_model_path = 'face_detection_yunet_2023mar.onnx'
detect_model_url = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'

# --- Path dan URL model Face Recognition (SFace) ---
recog_model_path = 'face_recognition_sface_2021dec.onnx'
recog_model_url = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'

# Unduh model secara otomatis jika belum ada di direktori
for path, dl_url in [(detect_model_path, detect_model_url), (recog_model_path, recog_model_url)]:
    if not os.path.exists(path):
        print(f"Mengunduh model {path}...")
        urllib.request.urlretrieve(dl_url, path)
        print("Model berhasil diunduh.")

# Inisialisasi FaceDetectorYN (deteksi wajah)
face_detector = cv2.FaceDetectorYN_create(
    model=detect_model_path,
    config='',
    input_size=(800, 600),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

# Detector khusus untuk enrollment: threshold lebih longgar karena foto referensi
# biasanya sudah bersih/terkontrol (bukan live feed dengan background ramai),
# jadi aman untuk lebih permisif supaya foto dengan angle sedikit ekstrem tetap kedeteksi.
enroll_detector = cv2.FaceDetectorYN_create(
    model=detect_model_path,
    config='',
    input_size=(800, 600),
    score_threshold=0.3,
    nms_threshold=0.3,
    top_k=5000
)

# Inisialisasi FaceRecognizerSF (pengenalan wajah)
recognizer = cv2.FaceRecognizerSF_create(model=recog_model_path, config='')


def load_image_correct_orientation(path, max_dim=800):
    """
    Baca gambar dengan koreksi EXIF orientation dan resize ke resolusi optimal (max_dim=800px)
    agar deteksi landmark YuNet presisi dan tidak menghasilkan crop wajah yang terdistorsi.
    """
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img)  # otomatis putar sesuai EXIF
    pil_img = pil_img.convert('RGB')
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Resize gambar besar agar receptive field YuNet presisi membaca landmark wajah
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return img


def get_embedding(image, face_row):
    """Align + crop wajah dari 1 baris hasil deteksi YuNet, lalu ekstrak embedding-nya."""
    aligned_face = recognizer.alignCrop(image, face_row)
    return recognizer.feature(aligned_face)


def enroll_known_faces():
    """
    Baca semua foto di known_faces/<nama>/*.jpg|png, deteksi wajahnya,
    lalu simpan embedding-nya ke database (list of (nama, embedding)).
    """
    database = []

    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"Folder '{KNOWN_FACES_DIR}' belum ada, dibuat otomatis. "
              f"Isi dengan subfolder per orang (mis. known_faces/Budi/foto1.jpg).")
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        return database

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
                print(f"  [!] Gagal baca {photo_path} ({e}), dilewati.")
                continue

            h, w, _ = img.shape
            enroll_detector.setInputSize((w, h))
            _, faces = enroll_detector.detect(img)

            if faces is None or len(faces) == 0:
                print(f"  [!] {os.path.basename(photo_path)}: TIDAK ADA wajah terdeteksi.")
                continue

            # Ambil wajah dengan confidence tertinggi kalau ada lebih dari satu
            best_face = max(faces, key=lambda f: f[-1])
            score = best_face[-1]
            embedding = get_embedding(img, best_face)
            database.append((person_name, embedding))
            count += 1
            print(f"  [OK] {os.path.basename(photo_path)}: terdeteksi ({w}x{h}), confidence={score:.3f}")

        print(f"  Enrolled '{person_name}': {count} foto berhasil diproses dari {len(photo_paths)} foto.")

    return database


def recognize(embedding, database):
    """Bandingkan 1 embedding wajah ke seluruh database, kembalikan (nama, skor) terbaik."""
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


def main():
    print("Enrolling known faces dari folder referensi...")
    known_faces_db = enroll_known_faces()
    print(f"Total {len(known_faces_db)} embedding wajah ter-enroll dari {len(set(n for n, _ in known_faces_db))} orang.\n")

    print("Mulai menarik gambar dari ESP32-CAM...")
    print("Tekan 'q' pada jendela tampilan (GUI) ATAU di terminal untuk keluar.")

    while True:
        try:
            if sys.platform == 'win32' and msvcrt.kbhit():
                if msvcrt.getch().lower() == b'q':
                    print("\nTombol 'q' ditekan di terminal. Menutup program...")
                    break

            # 1. Menarik gambar satu per satu dari endpoint /capture (dengan timeout 3 detik)
            with urllib.request.urlopen(url, timeout=3) as img_resp:
                imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)

            # 2. Decode data array menjadi frame gambar (format OpenCV)
            frame = cv2.imdecode(imgnp, -1)
            if frame is None:
                continue

            # 3. Sesuaikan input_size detector dengan ukuran frame yang diterima
            h, w, _ = frame.shape
            face_detector.setInputSize((w, h))

            # 4. Proses Face Detection menggunakan YuNet
            _, faces = face_detector.detect(frame)

            # 5. Untuk tiap wajah terdeteksi: recognize lalu gambar kotak + label
            if faces is not None and len(faces) > 0:
                global _last_sent_name, _last_sent_time
                for face in faces:
                    x, y, fw, fh = map(int, face[:4])
                    x, y = max(0, x), max(0, y)
                    fw, fh = min(fw, w - x), min(fh, h - y)

                    # --- FACE RECOGNITION ---
                    embedding = get_embedding(frame, face)
                    name, score = recognize(embedding, known_faces_db)

                    # Kirim notifikasi ke Serial Monitor ESP32-CAM (cooldown 1.5 detik atau jika nama berganti)
                    now = time.time()
                    if (now - _last_sent_time > 1.5) or (name != _last_sent_name):
                        send_face_to_esp32_async(IP_ADDRESS, name, score)
                        _last_sent_name = name
                        _last_sent_time = now

                    # Kirim log absensi ke Dashboard Django (wajah recognized & unrecognized)
                    if ENABLE_DASHBOARD:
                        global _dashboard_last_sent
                        face_status = "recognized" if name != "Unknown" else "unrecognized"
                        cooldown = DASHBOARD_COOLDOWN_RECOGNIZED if face_status == "recognized" else DASHBOARD_COOLDOWN_UNKNOWN
                        dash_key = (CAMERA_NAME, name if face_status == "recognized" else "Unknown")
                        last_dash_time = _dashboard_last_sent.get(dash_key, 0)

                        if now - last_dash_time > cooldown:
                            face_crop = crop_face_roi(frame, face)
                            send_log_to_dashboard_async(DASHBOARD_URL, CAMERA_NAME, name, score, face_status, face_crop)
                            _dashboard_last_sent[dash_key] = now

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

                    label = f"{name} ({score:.2f})"
                    cv2.putText(frame, label, (x, max(0, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 6. Tampilkan hasil pemrosesan di jendela komputer
            cv2.imshow("ESP32-CAM Face Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            terminal_pressed_q = (sys.platform == 'win32' and msvcrt.kbhit() and msvcrt.getch().lower() == b'q')

            if key == ord('q') or terminal_pressed_q:
                print("\nKeluar dari program...")
                break

        except KeyboardInterrupt:
            print("\nProgram dihentikan oleh pengguna (Ctrl+C).")
            break
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"Gagal mengambil gambar dari ESP32-CAM ({e}). Mencoba lagi...")
            if sys.platform == 'win32' and msvcrt.kbhit():
                if msvcrt.getch().lower() == b'q':
                    print("\nTombol 'q' ditekan di terminal. Menutup program...")
                    break
            continue
        except Exception as e:
            print(f"Terjadi kesalahan: {e}")
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
