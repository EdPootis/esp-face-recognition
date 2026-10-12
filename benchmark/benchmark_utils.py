"""
==========================================================================
  BENCHMARK UTILS - Modul Utilitas Bersama untuk Benchmark Face AI Models
==========================================================================
Menyediakan:
  - CameraStreamer (koneksi ESP32-S3 via MJPEG stream)
  - MetricsCollector (pengumpulan metrik latensi, FPS, akurasi)
  - GuidedScenarioRunner (pemandu skenario tes langkah demi langkah)
  - Fungsi logging hasil ke terminal & file .txt

Target Deployment: Jetson Nano / Raspberry Pi 3
Sumber Video: ESP32-S3 MJPEG Stream (http://IP:81/stream)
"""

import os
import sys
import time
import json
import http.client
import urllib.parse
import threading
import argparse
import platform
import subprocess
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==============================================================================
# KONSTANTA & KONFIGURASI DEFAULT
# ==============================================================================
DEFAULT_STREAM_URL = "http://10.7.28.33:81/stream"
DEFAULT_BENCHMARK_DURATION = 60  # detik per skenario
DEFAULT_DET_SIZE = (320, 240)
KNOWN_FACES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'known_faces')
BENCHMARK_LOG_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Skenario benchmark guided
GUIDED_SCENARIOS = [
    {
        "id": "frontal",
        "name": "Wajah Frontal (Hadap Depan)",
        "instruction": "Hadapkan wajah Anda LURUS ke kamera, jarak ~50-60cm.",
        "icon": "👤",
        "focus": "Baseline akurasi deteksi & recognition pada kondisi ideal"
    },
    {
        "id": "tilt_left",
        "name": "Miring Kiri (~30-45°)",
        "instruction": "Miringkan kepala Anda ke KIRI sekitar 30-45 derajat.",
        "icon": "↩️",
        "focus": "Robustness terhadap rotasi wajah (yaw kiri)"
    },
    {
        "id": "tilt_right",
        "name": "Miring Kanan (~30-45°)",
        "instruction": "Miringkan kepala Anda ke KANAN sekitar 30-45 derajat.",
        "icon": "↪️",
        "focus": "Robustness terhadap rotasi wajah (yaw kanan)"
    },
    {
        "id": "far_distance",
        "name": "Jarak Jauh (~1.5-2 meter)",
        "instruction": "Mundur dari kamera hingga jarak ~1.5-2 meter.",
        "icon": "📏",
        "focus": "Kemampuan deteksi small-face (simulasi kamera plafon/dinding)"
    },
    {
        "id": "low_light",
        "name": "Pencahayaan Redup / Backlight",
        "instruction": "Kurangi pencahayaan ruangan ATAU berdiri membelakangi sumber cahaya (backlight).",
        "icon": "🌙",
        "focus": "Ketahanan terhadap variasi pencahayaan / backlight"
    },
]


# ==============================================================================
# CAMERA STREAMER (MJPEG dari ESP32-S3)
# ==============================================================================
class CameraStreamer(threading.Thread):
    """
    Thread yang membaca MJPEG stream dari ESP32-S3 dan menyimpan frame terbaru.
    Mendukung mode 'stream' (port 81 /stream) dan 'capture' (port 80 /capture).
    """
    def __init__(self, url, timeout=5.0):
        super().__init__(daemon=True)
        self.url = url
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
        self.path = parsed.path if parsed.path else '/stream'
        if parsed.query:
            self.path += '?' + parsed.query
        self.conn = None
        self.mode = 'stream' if '/stream' in self.path else 'capture'

    def _get_connection(self):
        if self.conn is None:
            self.conn = http.client.HTTPConnection(self.host, timeout=self.timeout)
        return self.conn

    def _run_stream(self):
        conn = self._get_connection()
        conn.request('GET', self.path, headers={'User-Agent': 'BenchmarkClient'})
        resp = conn.getresponse()

        if resp.status != 200:
            resp.read()
            with self.lock:
                self.connected = False
            time.sleep(0.5)
            return

        bytes_buffer = b''
        while self.running:
            chunk = resp.read(4096)
            if not chunk:
                break
            bytes_buffer += chunk

            a = bytes_buffer.find(b'\xff\xd8')
            b = bytes_buffer.find(b'\xff\xd9', a) if a != -1 else -1

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
        conn.request('GET', self.path, headers={
            'User-Agent': 'BenchmarkClient',
            'Connection': 'keep-alive'
        })
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
# VIDEO FILE STREAMER (Untuk Benchmark yang Adil & Reproducible)
# ==============================================================================
class VideoFileStreamer(threading.Thread):
    """
    Thread yang membaca frame dari file video (MP4/AVI).
    Mensimulasikan live stream dengan membaca frame sesuai framerate asli video.
    """
    def __init__(self, video_path, override_fps=None):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.override_fps = override_fps
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.connected = False
        self.fps = 0.0
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            with self.lock:
                self.connected = False
            return
            
        with self.lock:
            self.connected = True
            
        # Gunakan FPS override jika diberikan (sangat berguna untuk video time-lapse dari ESP32)
        if self.override_fps is not None and self.override_fps > 0:
            video_fps = self.override_fps
        else:
            video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0 or np.isnan(video_fps):
                video_fps = 15.0  # fallback
                
        frame_delay = 1.0 / video_fps
        self.fps = video_fps
        
        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()
            
            if not ret:
                # Loop video jika sudah habis
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            with self.lock:
                self.frame = frame
                
            # Simulasi delay sesuai FPS
            elapsed = time.time() - start_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def get_total_frames(self):
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0
                
    def get_latest_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy(), self.connected, self.fps
            return None, self.connected, self.fps

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ==============================================================================
# METRICS COLLECTOR
# ==============================================================================
class MetricsCollector:
    """
    Mengumpulkan metrik performa untuk satu pasangan model FD+FR.
    Metrik yang dikumpulkan per skenario:
      - Detection latency (ms) per frame
      - Recognition latency (ms) per frame
      - Total pipeline latency (ms) per frame
      - Detection rate (wajah terdeteksi / total frame)
      - Recognition rate (wajah dikenali / wajah terdeteksi)
      - FPS throughput
      - Confidence scores (detection & recognition)
    """
    def __init__(self, pair_name):
        self.pair_name = pair_name
        self.scenarios = {}
        self.current_scenario = None
        self.start_time = None

    def begin_scenario(self, scenario_id):
        self.current_scenario = scenario_id
        self.scenarios[scenario_id] = {
            "detection_latency_ms": [],
            "recognition_latency_ms": [],
            "total_pipeline_latency_ms": [],
            "detection_confidence": [],
            "recognition_score": [],
            "faces_detected_count": 0,
            "faces_recognized_count": 0,
            "frames_processed": 0,
            "recognized_names": defaultdict(int),
            "fps_samples": [],
            "start_time": time.time(),
            "end_time": None,
        }

    def end_scenario(self):
        if self.current_scenario and self.current_scenario in self.scenarios:
            self.scenarios[self.current_scenario]["end_time"] = time.time()
        self.current_scenario = None

    def record_frame(self, det_latency_ms, rec_latency_ms, total_latency_ms,
                     face_detected, face_recognized, det_confidence=0.0,
                     rec_score=0.0, rec_name="Unknown", stream_fps=0.0):
        if self.current_scenario is None:
            return

        s = self.scenarios[self.current_scenario]
        s["frames_processed"] += 1
        s["detection_latency_ms"].append(det_latency_ms)
        s["total_pipeline_latency_ms"].append(total_latency_ms)

        if stream_fps > 0:
            s["fps_samples"].append(stream_fps)

        if face_detected:
            s["faces_detected_count"] += 1
            s["detection_confidence"].append(det_confidence)
            s["recognition_latency_ms"].append(rec_latency_ms)

            if face_recognized:
                s["faces_recognized_count"] += 1
                s["recognition_score"].append(rec_score)
                s["recognized_names"][rec_name] += 1

    def get_scenario_summary(self, scenario_id):
        s = self.scenarios.get(scenario_id)
        if s is None or s["frames_processed"] == 0:
            return None

        elapsed = (s["end_time"] or time.time()) - s["start_time"]

        def safe_stats(arr):
            if not arr:
                return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "p95": 0.0, "p99": 0.0}
            arr_np = np.array(arr)
            return {
                "mean": float(np.mean(arr_np)),
                "median": float(np.median(arr_np)),
                "min": float(np.min(arr_np)),
                "max": float(np.max(arr_np)),
                "std": float(np.std(arr_np)),
                "p95": float(np.percentile(arr_np, 95)),
                "p99": float(np.percentile(arr_np, 99)),
            }

        total_frames = s["frames_processed"]
        detected = s["faces_detected_count"]
        recognized = s["faces_recognized_count"]

        return {
            "scenario_id": scenario_id,
            "elapsed_seconds": round(elapsed, 2),
            "frames_processed": total_frames,
            "effective_fps": round(total_frames / elapsed, 2) if elapsed > 0 else 0.0,
            "stream_fps": safe_stats(s["fps_samples"]),
            "detection_rate": round(detected / total_frames * 100, 2) if total_frames > 0 else 0.0,
            "recognition_rate": round(recognized / detected * 100, 2) if detected > 0 else 0.0,
            "detection_latency": safe_stats(s["detection_latency_ms"]),
            "recognition_latency": safe_stats(s["recognition_latency_ms"]),
            "total_pipeline_latency": safe_stats(s["total_pipeline_latency_ms"]),
            "detection_confidence": safe_stats(s["detection_confidence"]),
            "recognition_score": safe_stats(s["recognition_score"]),
            "recognized_names": dict(s["recognized_names"]),
        }

    def get_full_report(self):
        report = {
            "pair_name": self.pair_name,
            "timestamp": datetime.now().isoformat(),
            "scenarios": {}
        }
        for sid in self.scenarios:
            summary = self.get_scenario_summary(sid)
            if summary:
                report["scenarios"][sid] = summary
        return report


# ==============================================================================
# GUIDED SCENARIO RUNNER
# ==============================================================================
class GuidedScenarioRunner:
    """
    Menjalankan skenario benchmark secara berurutan dengan panduan visual
    di terminal. Menghitung countdown dan memberi instruksi ke pengguna.
    """

    @staticmethod
    def print_header(pair_name, device_info):
        w = 72
        print("\n" + "=" * w)
        print(f"{'FACE AI MODEL BENCHMARK':^{w}}")
        print(f"{'Perbandingan Pasangan Model FD + FR':^{w}}")
        print("=" * w)
        print(f"  Pasangan Model  : {pair_name}")
        print(f"  Device          : {device_info}")
        print(f"  Waktu Mulai     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Total Skenario  : {len(GUIDED_SCENARIOS)}")
        print("=" * w)

    @staticmethod
    def print_scenario_start(scenario, scenario_num, total, duration):
        w = 68
        print("\n" + "─" * w)
        print(f"  {scenario['icon']}  SKENARIO {scenario_num}/{total}: {scenario['name']}")
        print(f"  📋 Instruksi: {scenario['instruction']}")
        print(f"  🎯 Fokus    : {scenario['focus']}")
        print(f"  ⏱️  Durasi   : {duration} detik")
        print("─" * w)

    @staticmethod
    def print_countdown(seconds):
        print(f"  Mulai dalam: ", end="", flush=True)
        for i in range(seconds, 0, -1):
            print(f"{i}.. ", end="", flush=True)
            time.sleep(1)
        print("MULAI!")

    @staticmethod
    def print_progress(elapsed, duration, frames, detected, recognized, fps):
        pct = min(100, int(elapsed / duration * 100))
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        remaining = max(0, duration - elapsed)

        line = (
            f"\r  [{bar}] {pct:3d}% | "
            f"Sisa: {remaining:5.1f}s | "
            f"Frame: {frames:4d} | "
            f"Det: {detected:4d} | "
            f"Rec: {recognized:4d} | "
            f"FPS: {fps:5.1f}"
        )
        print(line, end="", flush=True)

    @staticmethod
    def print_scenario_result(summary):
        if summary is None:
            print("  ⚠️  Tidak ada data untuk skenario ini.")
            return

        w = 60
        print(f"\n\n  {'─' * w}")
        print(f"  📊 HASIL SKENARIO: {summary['scenario_id'].upper()}")
        print(f"  {'─' * w}")
        print(f"  Frame Diproses       : {summary['frames_processed']}")
        print(f"  Effective FPS        : {summary['effective_fps']:.2f}")
        print(f"  Detection Rate       : {summary['detection_rate']:.1f}%")
        print(f"  Recognition Rate     : {summary['recognition_rate']:.1f}%")
        print(f"  ")
        print(f"  Latency Deteksi (ms) : "
              f"mean={summary['detection_latency']['mean']:.1f}  "
              f"median={summary['detection_latency']['median']:.1f}  "
              f"P95={summary['detection_latency']['p95']:.1f}  "
              f"P99={summary['detection_latency']['p99']:.1f}")
        print(f"  Latency Recogn. (ms) : "
              f"mean={summary['recognition_latency']['mean']:.1f}  "
              f"median={summary['recognition_latency']['median']:.1f}  "
              f"P95={summary['recognition_latency']['p95']:.1f}  "
              f"P99={summary['recognition_latency']['p99']:.1f}")
        print(f"  Latency Pipeline (ms): "
              f"mean={summary['total_pipeline_latency']['mean']:.1f}  "
              f"median={summary['total_pipeline_latency']['median']:.1f}  "
              f"P95={summary['total_pipeline_latency']['p95']:.1f}  "
              f"P99={summary['total_pipeline_latency']['p99']:.1f}")

        if summary['detection_confidence']['mean'] > 0:
            print(f"  Confidence Deteksi   : "
                  f"mean={summary['detection_confidence']['mean']:.3f}  "
                  f"min={summary['detection_confidence']['min']:.3f}  "
                  f"max={summary['detection_confidence']['max']:.3f}")
        if summary['recognition_score']['mean'] > 0:
            print(f"  Skor Recognition     : "
                  f"mean={summary['recognition_score']['mean']:.3f}  "
                  f"min={summary['recognition_score']['min']:.3f}  "
                  f"max={summary['recognition_score']['max']:.3f}")

        if summary['recognized_names']:
            top = sorted(summary['recognized_names'].items(), key=lambda x: -x[1])[:5]
            names_str = ", ".join([f"{n}({c}x)" for n, c in top])
            print(f"  Nama Terkenali       : {names_str}")


# ==============================================================================
# LOGGING & REPORT
# ==============================================================================
def save_report(report, pair_tag):
    """Simpan laporan ke file JSON & TXT di folder results/."""
    os.makedirs(BENCHMARK_LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON detail
    json_path = os.path.join(BENCHMARK_LOG_DIR, f"benchmark_{pair_tag}_{ts}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Laporan JSON disimpan: {json_path}")

    # TXT ringkasan (human-readable)
    txt_path = os.path.join(BENCHMARK_LOG_DIR, f"benchmark_{pair_tag}_{ts}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 72}\n")
        f.write(f"  BENCHMARK REPORT: {report['pair_name']}\n")
        f.write(f"  Timestamp: {report['timestamp']}\n")
        if 'device_info' in report:
            f.write(f"  Device: {report['device_info']}\n")
        if 'model_info' in report:
            mi = report['model_info']
            f.write(f"  Detection Model : {mi.get('detection_model', 'N/A')}\n")
            f.write(f"  Detection Size  : {mi.get('detection_model_size_bytes', 'N/A')} bytes\n")
            f.write(f"  Recognition Model: {mi.get('recognition_model', 'N/A')}\n")
            f.write(f"  Recognition Size : {mi.get('recognition_model_size_bytes', 'N/A')} bytes\n")
            f.write(f"  Backend/Target  : {mi.get('backend', 'N/A')}\n")
        f.write(f"{'=' * 72}\n\n")

        for sid, summary in report.get("scenarios", {}).items():
            sc_name = sid.upper()
            for gs in GUIDED_SCENARIOS:
                if gs["id"] == sid:
                    sc_name = gs["name"]
                    break
            f.write(f"--- SKENARIO: {sc_name} ---\n")
            f.write(f"  Durasi              : {summary['elapsed_seconds']:.1f}s\n")
            f.write(f"  Frame Diproses      : {summary['frames_processed']}\n")
            f.write(f"  Effective FPS       : {summary['effective_fps']:.2f}\n")
            f.write(f"  Detection Rate      : {summary['detection_rate']:.1f}%\n")
            f.write(f"  Recognition Rate    : {summary['recognition_rate']:.1f}%\n")
            f.write(f"  Latency Deteksi     : mean={summary['detection_latency']['mean']:.1f}ms  "
                    f"median={summary['detection_latency']['median']:.1f}ms  "
                    f"P95={summary['detection_latency']['p95']:.1f}ms  "
                    f"P99={summary['detection_latency']['p99']:.1f}ms\n")
            f.write(f"  Latency Recogn.     : mean={summary['recognition_latency']['mean']:.1f}ms  "
                    f"median={summary['recognition_latency']['median']:.1f}ms  "
                    f"P95={summary['recognition_latency']['p95']:.1f}ms  "
                    f"P99={summary['recognition_latency']['p99']:.1f}ms\n")
            f.write(f"  Latency Pipeline    : mean={summary['total_pipeline_latency']['mean']:.1f}ms  "
                    f"median={summary['total_pipeline_latency']['median']:.1f}ms  "
                    f"P95={summary['total_pipeline_latency']['p95']:.1f}ms  "
                    f"P99={summary['total_pipeline_latency']['p99']:.1f}ms\n")

            if summary['detection_confidence']['mean'] > 0:
                f.write(f"  Confidence Deteksi  : mean={summary['detection_confidence']['mean']:.3f}  "
                        f"min={summary['detection_confidence']['min']:.3f}  "
                        f"max={summary['detection_confidence']['max']:.3f}\n")
            if summary['recognition_score']['mean'] > 0:
                f.write(f"  Skor Recognition    : mean={summary['recognition_score']['mean']:.3f}  "
                        f"min={summary['recognition_score']['min']:.3f}  "
                        f"max={summary['recognition_score']['max']:.3f}\n")

            if summary['recognized_names']:
                top = sorted(summary['recognized_names'].items(), key=lambda x: -x[1])[:5]
                names_str = ", ".join([f"{n}({c}x)" for n, c in top])
                f.write(f"  Nama Terkenali      : {names_str}\n")
            f.write("\n")

    print(f"  📄 Laporan TXT disimpan: {txt_path}")
    return json_path, txt_path


def get_device_info():
    """Mendeteksi informasi perangkat secara otomatis."""
    info = {
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "processor": platform.processor() or "Unknown",
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
    }

    # Deteksi Jetson Nano
    if os.path.exists('/proc/device-tree/model'):
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip('\x00').strip()
                info["board_model"] = model
                if "jetson" in model.lower():
                    info["device_type"] = "Jetson Nano"
        except Exception:
            pass

    # Deteksi Raspberry Pi
    if os.path.exists('/proc/cpuinfo'):
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo:
                    info["device_type"] = "Raspberry Pi"
                    for line in cpuinfo.split('\n'):
                        if 'Model' in line:
                            info["board_model"] = line.split(':')[-1].strip()
                            break
        except Exception:
            pass

    # RAM info
    if sys.platform == 'linux':
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        mem_kb = int(line.split()[1])
                        info["ram_total_mb"] = round(mem_kb / 1024)
                        break
        except Exception:
            pass
    elif sys.platform == 'win32':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', c_ulonglong),
                    ('ullAvailPhys', c_ulonglong),
                    ('ullTotalPageFile', c_ulonglong),
                    ('ullAvailPageFile', c_ulonglong),
                    ('ullTotalVirtual', c_ulonglong),
                    ('ullAvailVirtual', c_ulonglong),
                    ('ullAvailExtendedVirtual', c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            info["ram_total_mb"] = round(stat.ullTotalPhys / (1024 * 1024))
        except Exception:
            pass

    # CUDA info
    try:
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        info["cuda_devices"] = cuda_count
        if cuda_count > 0:
            info["cuda_available"] = True
    except Exception:
        info["cuda_available"] = False

    if "device_type" not in info:
        if sys.platform == 'win32':
            info["device_type"] = "Windows PC/Laptop"
        else:
            info["device_type"] = "Linux PC/Server"

    return info


def get_model_file_size(path):
    """Mendapatkan ukuran file model dalam bytes."""
    if os.path.exists(path):
        return os.path.getsize(path)
    return 0


def create_base_argparser(pair_name):
    """Membuat argparse standar untuk semua script benchmark."""
    parser = argparse.ArgumentParser(
        description=f"Benchmark Pasangan Model: {pair_name}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--stream-url', type=str, default=DEFAULT_STREAM_URL,
        help=f"URL MJPEG stream ESP32-S3 (Default: {DEFAULT_STREAM_URL})"
    )
    parser.add_argument(
        '--video-file', type=str, default=None,
        help="Path ke file video lokal (.mp4) untuk input agar pengujian bisa diulang secara identik. Jika diset, stream-url akan diabaikan."
    )
    parser.add_argument(
        '--video-fps', type=float, default=None,
        help="Timpa FPS video rekaman. Berguna jika video ESP32 merekam terlalu cepat. Contoh: 4.5"
    )
    parser.add_argument(
        '--duration', type=int, default=DEFAULT_BENCHMARK_DURATION,
        help=f"Durasi benchmark per skenario dalam detik (Default: {DEFAULT_BENCHMARK_DURATION})"
    )
    parser.add_argument(
        '--det-width', type=int, default=DEFAULT_DET_SIZE[0],
        help=f"Lebar input deteksi (Default: {DEFAULT_DET_SIZE[0]})"
    )
    parser.add_argument(
        '--det-height', type=int, default=DEFAULT_DET_SIZE[1],
        help=f"Tinggi input deteksi (Default: {DEFAULT_DET_SIZE[1]})"
    )
    parser.add_argument(
        '--no-cuda', action='store_true',
        help="Paksa gunakan CPU (nonaktifkan CUDA)"
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help="Jalankan tanpa jendela OpenCV (mode headless)"
    )
    parser.add_argument(
        '--threshold', type=float, default=None,
        help="Override threshold recognition (default tergantung model)"
    )
    parser.add_argument(
        '--countdown', type=int, default=5,
        help="Countdown sebelum setiap skenario dalam detik (Default: 5)"
    )
    return parser


def format_device_string(info):
    """Format informasi device menjadi satu baris string."""
    parts = [info.get("device_type", "Unknown")]
    if "board_model" in info:
        parts.append(info["board_model"])
    if "ram_total_mb" in info:
        parts.append(f"RAM {info['ram_total_mb']}MB")
    parts.append(f"OpenCV {info.get('opencv_version', '?')}")
    if info.get("cuda_available"):
        parts.append(f"CUDA ({info.get('cuda_devices', 0)} device)")
    return " | ".join(parts)
