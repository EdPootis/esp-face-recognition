"""
Presensi KTM + Foto Dokumentasi (mode eksperimen, turunan ktm_face_verification_optimized.py).

Tap KTM langsung dicatat sebagai kehadiran, seperti Presentronik lama. Kamera hanya dipakai untuk
dokumentasi: satu foto diambil dari /capture saat kartu di-tap. Tidak ada face recognition, dan
presensi tetap dicatat walau foto gagal diambil.

Alur per tap (firmware ktm_foto_s3 -> GET /tap?uid=..&cam=..&mac=..):
  1. Cari pemilik kartu di students_database.json.
  2. Ambil foto JPEG dari http://<ip-kamera>/capture.
  3. Simpan foto di FOTO_DIR (cadangan lokal).
  4. Kirim nama ke LCD ESP32 (/face).
  5. Kirim log presensi + foto ke dashboard (POST /api/log/).

GUI menampilkan satu tile per kamera: foto tap terakhir (atau alasan foto tidak ada) dan status
LCD & dashboard. GUI butuh OpenCV; dengan --headless, atau tanpa OpenCV/DISPLAY, skrip hanya memakai
library standar Python. Tetap kompatibel Python 3.6 (Jetson Nano). Uji tanpa hardware: tekan 't'
di GUI, atau
  curl "http://localhost:5050/tap?uid=DEA35B89&cam=ESP32-S3-1"
"""
import os
import re
import sys
import time
import json
import base64
import argparse
import textwrap
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

try:
    import cv2
    import numpy as np
except ImportError:  # Mode headless tetap jalan tanpa OpenCV
    cv2 = None

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==============================================================================
# KONFIGURASI PRESENSI KTM + FOTO DOKUMENTASI
# ==============================================================================

# Daftar IP Address ESP32-S3. Kunci harus sama persis dengan NODE_NAME di firmware.
# Fallback jika dashboard (GET /api/kamera/) tidak bisa dihubungi atau belum punya kamera;
# saat start daftar ini diganti dengan daftar kamera dari dashboard (lihat load_camera_hosts).
CAMERA_HOSTS = {
    'ESP32-S3-1': '10.7.28.33',
    'ESP32-S3-2': '10.7.28.47',
}

# Database Pemetaan UID KTM -> Mahasiswa (sama dengan skrip face recognition)
STUDENTS_DB_FILE = 'students_database.json'

# Port HTTP Webhook Server (Untuk menerima notifikasi tap kartu dari ESP32 via WiFi)
TAP_SERVER_PORT = 5050

# Pengambilan foto dari /capture ESP32. Total waktu terburuk ~ (1 + CAPTURE_RETRIES) x CAPTURE_TIMEOUT,
# dan LCD baru dibalas setelahnya, jadi jaga agar tetap di bawah RESPONSE_TIMEOUT_MS firmware (5 detik).
CAPTURE_TIMEOUT = 2.0
CAPTURE_RETRIES = 1

# Salinan lokal foto presensi: FOTO_DIR/<tanggal>/<jam>_<milidetik>_<kamera>_<NIM atau UID>.jpg
SAVE_LOCAL_PHOTOS = True
FOTO_DIR = 'foto_presensi'

# Pesan LCD (maks 16 karakter, baris 2) untuk kartu yang tidak ada di database
LCD_UNKNOWN_CARD = 'KTM TDK DIKENAL'

# Konfigurasi Dashboard Django (Log Absensi)
DASHBOARD_URL = 'http://10.7.28.21:8000/api/log/'
ENABLE_DASHBOARD = True
DASHBOARD_API_KEY = 'presentronik_fr_secret_key_2026'
# Daftar kamera aktif dikelola di dashboard (Setting Kamera); host sama dengan DASHBOARD_URL
CAMERA_API_URL = DASHBOARD_URL.replace('/api/log/', '/api/kamera/')
# Ikut tercatat di log_presentronik, sehingga presensi mode ini bisa dibedakan dari mode face recognition
TIPE_PERANGKAT = 'ESP32-S3 KTM + Foto'

# Ukuran tile GUI per kamera: area foto + banner status di bawahnya
TILE_WIDTH = 640
TILE_HEIGHT = 480
BANNER_HEIGHT = 120


def load_camera_hosts(fallback, ruangan=None):
    """Ambil daftar kamera dari dashboard (GET /api/kamera/); jika gagal atau kosong, pakai CAMERA_HOSTS lokal."""
    if not ENABLE_DASHBOARD:
        print("[DASHBOARD] Dashboard nonaktif, memakai CAMERA_HOSTS lokal.")
        return fallback
    url = CAMERA_API_URL
    if ruangan:
        url += '?' + urllib.parse.urlencode({'ruangan': ruangan})
    try:
        req = urllib.request.Request(url, headers={'X-API-Key': DASHBOARD_API_KEY, 'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            hosts = json.loads(resp.read().decode('utf-8')).get('camera_hosts') or {}
        if hosts:
            print(f"[DASHBOARD] {len(hosts)} kamera dari dashboard: {hosts}")
            return hosts
        filter_txt = f" di ruangan '{ruangan}'" if ruangan else ""
        print(f"[DASHBOARD WARN] Belum ada kamera aktif dengan alamat IP{filter_txt} di dashboard, "
              f"memakai CAMERA_HOSTS lokal.")
    except Exception as e:
        print(f"[DASHBOARD WARN] Gagal mengambil daftar kamera dari {CAMERA_API_URL} ({e}), memakai CAMERA_HOSTS lokal.")
    return fallback


# ==============================================================================
# MANAJEMEN DATABASE MAHASISWA & PEMETAAN UID KTM
# ==============================================================================
class StudentDatabase:
    """
    Mengelola pemetaan antara UID Kartu KTM dengan Biodata Mahasiswa.
    Format disimpan dalam students_database.json (dipakai bersama skrip face recognition).
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
            print(f"[WARN] {self.json_path} tidak ditemukan: semua kartu akan tercatat sebagai tidak dikenal.")

    def get_student_by_uid(self, uid):
        clean_uid = uid.replace(" ", "").upper()
        return self.students.get(clean_uid, None)


student_db = StudentDatabase()

# MAC terakhir yang dikirim tiap node lewat /tap: {nama_kamera: mac}
camera_macs = {}

# Hasil tap terakhir per kamera untuk GUI: {nama_kamera: {...}}, diisi bertahap oleh process_tap
tap_results = {}
tap_results_lock = threading.Lock()


def update_tap_result(cam_name, **fields):
    """Perbarui hasil tap terakhir sebuah kamera (dipanggil dari thread process_tap)."""
    with tap_results_lock:
        tap_results.setdefault(cam_name, {}).update(fields)


# ==============================================================================
# FOTO DOKUMENTASI, LCD & DASHBOARD
# ==============================================================================
def capture_photo(host):
    """Ambil satu foto JPEG dari /capture ESP32. Mengembalikan (bytes JPEG, None) atau (None, alasan gagal)."""
    url = f"http://{host}/capture"
    last_error = None
    for attempt in range(1 + CAPTURE_RETRIES):
        if attempt:
            time.sleep(0.3)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'PresensiFotoClient'})
            with urllib.request.urlopen(req, timeout=CAPTURE_TIMEOUT) as resp:
                data = resp.read()
            if data[:2] == b'\xff\xd8':
                return data, None
            last_error = f"respons bukan JPEG ({len(data)} byte)"
        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code} {e.reason}"
            if e.code == 503:
                last_error += " (kamera masih standby)"
        except Exception as e:
            last_error = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
    return None, last_error


def save_photo_local(jpeg, tap_time, cam_name, ident):
    """Simpan salinan foto presensi ke FOTO_DIR/<tanggal>/; mengembalikan path file atau None."""
    try:
        folder = os.path.join(FOTO_DIR, tap_time.strftime('%Y-%m-%d'))
        os.makedirs(folder, exist_ok=True)
        safe = lambda s: re.sub(r'[^A-Za-z0-9_.-]+', '_', s)
        # Milidetik ikut di nama file agar dua tap dalam detik yang sama tidak saling menimpa
        stamp = tap_time.strftime('%H%M%S_%f')[:-3]
        path = os.path.join(folder, f"{stamp}_{safe(cam_name)}_{safe(ident)}.jpg")
        with open(path, 'wb') as f:
            f.write(jpeg)
        return path
    except Exception as e:
        print(f"[FOTO WARN] Gagal menyimpan foto lokal: {e}")
        return None


def send_result_to_lcd(host, text, cam_name, tap_time):
    """Mengirim hasil presensi ke ESP32 (/face) agar LCD 16x2 terupdate. Mengembalikan alasan gagal atau None."""
    try:
        query = urllib.parse.urlencode({
            'name': text,
            'score': '0.00',  # Tidak ada skor wajah di mode ini; parameter tetap dikirim sesuai kontrak /face
            'cam': cam_name,
            'time': tap_time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        req = urllib.request.Request(f"http://{host}/face?{query}", headers={'User-Agent': 'PresensiFotoClient'})
        with urllib.request.urlopen(req, timeout=1.5) as _:
            pass
        return None
    except Exception as e:
        print(f"[LCD WARN] Gagal mengirim hasil ke LCD {cam_name} ({host}): {e}")
        return f"{type(e).__name__}: {getattr(e, 'reason', e)}"


def send_log_to_dashboard(cam_name, student, uid, jpeg, tap_time, mac_address=None):
    """
    Mengirim log presensi ke Django Dashboard API (skema LogPresentronik, sama dengan skrip face recognition).
    Kartu terdaftar -> status HADIR (masuk log_presentronik & sinkron SIAK); kartu tidak dikenal -> status
    Unknown (hanya log_absensi dashboard). Foto ikut dikirim jika ada. Mengembalikan ringkasan hasil untuk GUI.
    """
    if not ENABLE_DASHBOARD:
        return "NONAKTIF"

    known = student is not None
    nim = student.get('nim', '-') if known else '-'
    name = student.get('name', '-') if known else 'Unknown'
    kd_org = student.get('kd_org', student.get('major', '01.00.12.01')) if known else '01.00.12.01'
    foto_b64 = f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode('utf-8')}" if jpeg else None

    payload = {
        # Format Standar Model LogPresentronik
        'no_identitas': nim,
        'nama': name,
        'rfid_tag': uid,
        'kd_org': kd_org,
        'tipe_perangkat': TIPE_PERANGKAT,
        'ts_presensi_device': tap_time.strftime('%Y-%m-%d %H:%M:%S'),

        # Atribut Tambahan untuk Monitoring & Audit
        'kamera': cam_name,
        'skor': 0.0,  # Tidak ada pencocokan wajah di mode ini
        'status': 'HADIR' if known else 'Unknown',
        'foto_b64': foto_b64,

        # Alias Kompatibilitas
        'nim': nim,
        'person_name': name,
        'similarity': 0.0,
        'recognized': known,  # Di mode ini artinya kartu dikenali, bukan wajah
        'timestamp': tap_time.isoformat(),
        'image_base64': foto_b64
    }
    # MAC asli dari firmware; tanpa MAC, dashboard memakai MAC kamera dari Setting Kamera
    if mac_address:
        payload['mac_address'] = mac_address

    try:
        req = urllib.request.Request(
            DASHBOARD_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json', 'User-Agent': 'PresensiFotoClient',
                     'X-API-Key': DASHBOARD_API_KEY},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            code = resp.getcode()
        print(f"[DASHBOARD] Log presensi terkirim ({code}) -> {name} ({nim}) | UID: {uid} | "
              f"Foto: {'ada' if jpeg else 'TIDAK ADA'}")
        return f"OK ({code})"
    except Exception as e:
        print(f"[DASHBOARD ERROR] Gagal mengirim log ({cam_name} / {name} / UID {uid}): {e}")
        return f"GAGAL: {type(e).__name__}: {getattr(e, 'reason', e)}"


def process_tap(uid, cam_name, mac=None):
    """Memproses satu tap KTM: presensi selalu dicatat, foto hanya pelengkap dokumentasi."""
    tap_time = datetime.now()
    clean_uid = uid.replace(" ", "").upper()
    student = student_db.get_student_by_uid(clean_uid)
    if mac:
        camera_macs[cam_name] = mac
    host = CAMERA_HOSTS.get(cam_name)

    # Hasil tap sebelumnya di kamera ini diganti; seq dipakai GUI untuk decode foto sekali per tap
    with tap_results_lock:
        seq = tap_results.get(cam_name, {}).get('seq', 0) + 1
        tap_results[cam_name] = {'seq': seq, 'time': tap_time, 'uid': clean_uid, 'student': student,
                                 'jpeg': None, 'foto': 'Mengambil foto...', 'lcd': '-', 'dashboard': '-'}

    if student:
        print(f"\n[TAP KTM] {cam_name} | UID: {clean_uid} -> {student.get('name')} ({student.get('nim')})")
    else:
        print(f"\n[TAP KTM] {cam_name} | UID: {clean_uid} -> KARTU TIDAK TERDAFTAR")

    jpeg = None
    if host is None:
        # Presensi tetap dicatat; hanya foto & balasan LCD yang tidak bisa dilakukan tanpa IP kamera
        reason = (f"Kamera '{cam_name}' tidak ada di CAMERA_HOSTS {list(CAMERA_HOSTS)}. "
                  f"Samakan NODE_NAME di secrets.h firmware dengan nama kamera di dashboard.")
        print(f"[TAP WARN] {reason} Presensi dicatat tanpa foto.")
        update_tap_result(cam_name, foto=reason, lcd='Tidak dikirim (IP kamera tidak diketahui)')
    else:
        jpeg, error = capture_photo(host)
        if jpeg:
            update_tap_result(cam_name, jpeg=jpeg, foto=f"OK ({len(jpeg) // 1024} KB)")
        else:
            print(f"[FOTO WARN] Gagal mengambil foto dari http://{host}/capture: {error}")
            update_tap_result(cam_name, foto=f"Gagal dari http://{host}/capture: {error}")

        # Balas LCD setelah foto diambil, agar mahasiswa tetap di depan kamera sampai namanya tampil
        lcd_error = send_result_to_lcd(host, student.get('name', '-') if student else LCD_UNKNOWN_CARD,
                                       cam_name, tap_time)
        update_tap_result(cam_name, lcd='OK' if lcd_error is None else f"GAGAL: {lcd_error}")

    if jpeg and SAVE_LOCAL_PHOTOS:
        ident = student.get('nim', clean_uid) if student else clean_uid
        path = save_photo_local(jpeg, tap_time, cam_name, ident)
        if path:
            print(f"[FOTO] Tersimpan: {path} ({len(jpeg) // 1024} KB)")

    update_tap_result(cam_name, dashboard='Mengirim...')
    dashboard_result = send_log_to_dashboard(cam_name, student, clean_uid, jpeg, tap_time, camera_macs.get(cam_name))
    update_tap_result(cam_name, dashboard=dashboard_result)


def trigger_card_tap(uid, cam_name, mac=None):
    """Memproses tap di thread terpisah agar webhook langsung membalas ke ESP32."""
    threading.Thread(target=process_tap, args=(uid, cam_name, mac), daemon=True).start()


# ==============================================================================
# HTTP SERVER UNTUK MENERIMA WEBHOOK TAP KARTU DARI ESP32
# ==============================================================================
class TapWebhookHandler(BaseHTTPRequestHandler):
    """
    Server HTTP lokal agar ESP32 dapat mengirim data tap kartu secara langsung:
    Contoh: GET http://[IP_KOMPUTER]:5050/tap?uid=DEA35B89&cam=ESP32-S3-1&mac=24:6F:28:AB:CD:EF
    """
    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        if parsed_url.path in ('/tap', '/api/tap'):
            query = urllib.parse.parse_qs(parsed_url.query)
            uid = query.get('uid', [None])[0]
            cam = query.get('cam', ['ESP32-S3'])[0]
            mac = query.get('mac', [None])[0]

            if uid:
                trigger_card_tap(uid, cam, mac)
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
                mac = data.get('mac')
                if uid:
                    trigger_card_tap(uid, cam, mac)
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


# ==============================================================================
# GUI: SATU TILE PER KAMERA (FOTO TAP TERAKHIR + STATUS)
# ==============================================================================
def put_text_fit(img, text, org, scale, color, thickness):
    """cv2.putText yang mengecilkan font jika teks lebih lebar dari gambar."""
    max_w = img.shape[1] - org[0] - 10
    text_w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    if text_w > max_w:
        scale *= max_w / text_w
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def status_color(text):
    """Hijau jika berhasil, kuning jika masih diproses, merah jika gagal."""
    if text.startswith('OK'):
        return (0, 255, 0)
    if text in ('-', 'NONAKTIF') or text.startswith(('Mengambil', 'Mengirim')):
        return (0, 230, 255)
    return (80, 80, 255)


def render_tile(cam_name, result, photo_cache):
    """Tile satu kamera: foto tap terakhir, atau alasan foto tidak ada, ditambah banner status di bawahnya."""
    tile = np.zeros((TILE_HEIGHT + BANNER_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
    photo_area = tile[:TILE_HEIGHT]
    banner = tile[TILE_HEIGHT:]
    host = CAMERA_HOSTS.get(cam_name)

    if not result:
        photo_area[:] = (30, 30, 30)
        put_text_fit(photo_area, "Menunggu tap KTM...", (20, TILE_HEIGHT // 2), 0.8, (200, 200, 200), 2)
        banner[:] = (45, 30, 20)
        put_text_fit(banner, f"[{cam_name}] STANDBY", (20, 40), 0.75, (255, 255, 255), 2)
        put_text_fit(banner, f"Kamera: http://{host}" if host else "Kamera tidak ada di CAMERA_HOSTS",
                     (20, 80), 0.55, (200, 200, 200), 1)
        return tile

    # Foto di-decode sekali per tap (seq), bukan setiap frame GUI
    img = None
    if result.get('jpeg'):
        cached = photo_cache.get(cam_name)
        if cached and cached[0] == result['seq']:
            img = cached[1]
        else:
            img = cv2.imdecode(np.frombuffer(result['jpeg'], dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)
            photo_cache[cam_name] = (result['seq'], img)

    foto_txt = result.get('foto', '-')
    if result.get('jpeg') and img is None:
        foto_txt = "Foto diterima tetapi tidak bisa dibaca (JPEG rusak)"
    if img is not None:
        photo_area[:] = img
    elif foto_txt.startswith('Mengambil'):
        photo_area[:] = (30, 30, 30)
        put_text_fit(photo_area, "MENGAMBIL FOTO...", (20, TILE_HEIGHT // 2), 0.9, (0, 230, 255), 2)
    else:
        # Alasan foto tidak ada ditampilkan besar di area foto
        photo_area[:] = (25, 25, 60)
        put_text_fit(photo_area, "TIDAK ADA FOTO", (20, 150), 1.0, (80, 80, 255), 2)
        for i, line in enumerate(textwrap.wrap(foto_txt, 50)[:6]):
            put_text_fit(photo_area, line, (20, 200 + i * 30), 0.6, (255, 255, 255), 1)
    cv2.putText(photo_area, f"{cam_name} | {host or 'IP tidak diketahui'}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    student = result.get('student')
    banner[:] = (20, 60, 20) if student else (20, 20, 70)
    status = "HADIR" if student else "KARTU TIDAK TERDAFTAR"
    who = f"{student.get('name', '-')} ({student.get('nim', '-')})" if student else "Unknown"
    put_text_fit(banner, f"[{cam_name}] {result['time'].strftime('%H:%M:%S')}  {status}", (20, 30),
                 0.75, (255, 255, 255), 2)
    put_text_fit(banner, f"{who} | UID: {result['uid']}", (20, 60), 0.6, (255, 255, 255), 1)
    # Alasan lengkap sudah tampil besar di area foto; di banner cukup ringkasannya
    foto_short = foto_txt if img is not None or foto_txt.startswith('Mengambil') else "TIDAK ADA (alasan di atas)"
    put_text_fit(banner, f"Foto: {foto_short}", (20, 85), 0.5, status_color(foto_short), 1)
    put_text_fit(banner, f"LCD: {result['lcd']}", (20, 105), 0.5, status_color(result['lcd']), 1)
    put_text_fit(banner, f"Dashboard: {result['dashboard']}", (330, 105), 0.5, status_color(result['dashboard']), 1)
    return tile


def build_grid(tiles):
    """Susun tile kamera dalam grid 2 kolom (1 kolom jika hanya satu kamera)."""
    cols = 1 if len(tiles) == 1 else 2
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        row += [np.zeros_like(tiles[0])] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def run_gui(server, sim_cam):
    """Webhook berjalan di thread latar; thread utama menggambar GUI sampai 'q' ditekan."""
    threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.5}, daemon=True).start()
    window_name = "Presensi KTM + Foto Dokumentasi"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    photo_cache = {}
    try:
        while True:
            with tap_results_lock:
                snapshot = {cam: dict(r) for cam, r in tap_results.items()}
            # Kamera dari CAMERA_HOSTS, ditambah kamera tak dikenal yang pernah mengirim tap (agar salah nama terlihat)
            cams = list(CAMERA_HOSTS) + [c for c in snapshot if c not in CAMERA_HOSTS]
            tiles = [render_tile(cam, snapshot.get(cam), photo_cache) for cam in (cams or ['-'])]
            cv2.imshow(window_name, build_grid(tiles))

            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == 27:
                print("Menghentikan program...")
                break
            elif key == ord('t'):
                # Simulasi tap kartu (tanpa hardware). Kamera yang standby akan membalas 503 sampai ada tap asli.
                sample_uids = list(student_db.students)
                if sample_uids:
                    print(f"\n[SIMULASI] Tap kartu contoh UID '{sample_uids[0]}' di {sim_cam}")
                    trigger_card_tap(sample_uids[0], sim_cam)
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna (Ctrl+C)...")
    finally:
        server.shutdown()
        cv2.destroyAllWindows()


# ==============================================================================
# MAIN PROGRAM
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Presensi KTM + Foto Dokumentasi (tanpa face recognition)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--port', type=int, default=TAP_SERVER_PORT, help="Port Webhook Tap Kartu (Default: 5050)")
    parser.add_argument('--ruangan', type=str, default=None,
                        help="Hanya pakai kamera di ruangan ini dari dashboard (mis. A6.02). Default: semua kamera aktif")
    parser.add_argument('--headless', action='store_true',
                        help="Tanpa GUI (otomatis jika OpenCV tidak terpasang atau tidak ada DISPLAY)")
    parser.add_argument('--cam', type=str, default='ESP32-S3-1',
                        help="Kamera tujuan simulasi tap kartu (tombol 't' di GUI) (Default: ESP32-S3-1)")
    return parser.parse_args()


def main():
    global CAMERA_HOSTS
    args = parse_args()

    # GUI hanya jika OpenCV ada dan tersedia layar (bukan SSH tanpa X11 di Linux/Jetson)
    has_display = sys.platform == 'win32' or os.environ.get('DISPLAY') is not None
    use_gui = not args.headless and cv2 is not None and has_display
    if not args.headless and cv2 is None:
        print("[INFO] OpenCV tidak terpasang, berjalan tanpa GUI.")

    print("=" * 68)
    print("  PRESENSI KTM + FOTO DOKUMENTASI (TANPA FACE RECOGNITION)")
    print(f"  Dashboard        : {DASHBOARD_URL if ENABLE_DASHBOARD else 'NONAKTIF'}")
    print(f"  Foto Lokal       : {os.path.abspath(FOTO_DIR) if SAVE_LOCAL_PHOTOS else 'NONAKTIF'}")
    print(f"  Mode Display     : {'GUI WINDOW (OpenCV HighGUI)' if use_gui else 'HEADLESS (console saja)'}")
    print("=" * 68)

    # Daftar kamera dari dashboard diambil sekali saat start
    CAMERA_HOSTS = load_camera_hosts(CAMERA_HOSTS, args.ruangan)
    print(f"Total Database: {len(student_db.students)} KTM terdaftar.")

    try:
        server = HTTPServer(('0.0.0.0', args.port), TapWebhookHandler)
    except Exception as e:
        print(f"[ERROR] Gagal memulai Tap Webhook Server di port {args.port}: {e}")
        sys.exit(1)

    print(f"\n✓ Tap Webhook Server aktif di port {args.port}. Menunggu tap KTM dari ESP32...")
    if use_gui:
        print("  • Tekan 't' pada GUI untuk simulasi tap kartu, 'q' untuk keluar.\n")
        try:
            run_gui(server, args.cam)
        finally:
            server.server_close()
            print("Selesai.")
        return

    print("  • Tekan Ctrl+C untuk keluar.\n")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna (Ctrl+C)...")
    finally:
        server.server_close()
        print("Selesai.")


if __name__ == '__main__':
    main()
