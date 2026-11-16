"""
SCRFD-500M (InsightFace buffalo_sc det_500m.onnx) sebagai detector wajah skrip presensi.

Antarmuka meniru cv2.FaceDetectorYN (YuNet): detect(img) -> (1, baris Nx15
[x, y, w, h, 5 landmark x/y, skor]) atau (1, None). Dengan begitu alignCrop() SFace,
penggambaran bbox, dan penskalaan koordinat di ktm_face_verification*.py tidak berubah.
Urutan landmark sama dengan YuNet (lihat benchmark/combo_pipeline.py).

Decode disalin dari benchmark/combo_pipeline.py SCRFDDetector, yang sudah diverifikasi sama
dengan InsightFace hingga 0.000 px. Verifikasi ulang jika diubah.
Hanya butuh numpy, OpenCV, dan onnxruntime. Kompatibel Python 3.6 (Jetson Nano).

Hasil benchmark (pasangan #5, SCRFD + SFace) dibanding YuNet + SFace: kepala miring 94-97%
(YuNet hanya mendeteksi 20-42% miring kiri), jarak jauh 75-77% (vs ~56%), cahaya redup ~75%
(vs 65-69%), Jetson 65 ms per frame (vs 84 ms).

Lisensi: model InsightFace dirilis untuk riset non-komersial. Pemakaiannya sebagai detector
default diputuskan penanggung jawab proyek pada 16 November 2026.
"""
import copy
import os
import shutil
import urllib.request
import zipfile

import cv2
import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRFD_MODEL_PATH = os.path.join(MODULE_DIR, 'scrfd_500m.onnx')
BUFFALO_SC_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip'

# Lokasi lain yang mungkin sudah berisi model yang sama (benchmark, cache InsightFace)
SCRFD_MODEL_CANDIDATES = [
    SCRFD_MODEL_PATH,
    os.path.join(MODULE_DIR, 'benchmark', 'models', 'scrfd_500m.onnx'),
    os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'buffalo_sc', 'det_500m.onnx'),
]


def ensure_scrfd_model():
    """Kembalikan path model SCRFD; unduh det_500m.onnx dari buffalo_sc.zip jika belum ada."""
    for path in SCRFD_MODEL_CANDIDATES:
        if os.path.exists(path):
            return path

    print(f"Mengunduh model SCRFD-500M (buffalo_sc.zip dari rilis InsightFace)...")
    zip_tmp = SCRFD_MODEL_PATH + '.zip.tmp'
    onnx_tmp = SCRFD_MODEL_PATH + '.tmp'
    try:
        urllib.request.urlretrieve(BUFFALO_SC_URL, zip_tmp)
        with zipfile.ZipFile(zip_tmp) as zf:
            member = next(n for n in zf.namelist() if n.endswith('det_500m.onnx'))
            with zf.open(member) as src, open(onnx_tmp, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        os.replace(onnx_tmp, SCRFD_MODEL_PATH)
    finally:
        for tmp in (zip_tmp, onnx_tmp):
            if os.path.exists(tmp):
                os.remove(tmp)
    print(f"Model {os.path.basename(SCRFD_MODEL_PATH)} berhasil diunduh.")
    return SCRFD_MODEL_PATH


def _create_ort_session(model_path, use_cuda):
    """Sama dengan create_ort_session di benchmark/combo_pipeline.py. Mengembalikan (session, provider)."""
    import onnxruntime as ort
    providers = []
    if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    so = ort.SessionOptions()
    so.log_severity_level = 3
    # Matikan spin-wait thread pool: tanpa ini, thread ORT terus berputar setelah run() dan
    # memperlambat SFace sesudahnya (diukur di benchmark hingga 3.4x)
    try:
        so.add_session_config_entry('session.intra_op.allow_spinning', '0')
    except Exception:
        pass
    session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    return session, session.get_providers()[0]


class SCRFDFaceDetector:
    """
    SCRFD-500M dengan antarmuka cv2.FaceDetectorYN. Ukuran input jaringan mengikuti gambar yang
    dideteksi (dibulatkan ke bawah ke kelipatan 32, rasio dipertahankan), sehingga perilakunya sama
    dengan YuNet di skrip: frame penuh di skrip dasar, frame 320x240 (-> 320x224, sama dengan
    benchmark #5) di skrip _optimized.
    Pre/post-processing = insightface/model_zoo/scrfd.py: resize mempertahankan rasio (pojok kiri
    atas), normalisasi (x-127.5)/128, 3 stride x 2 anchor, NMS 0.4.
    """
    nms_th = 0.4
    strides = (8, 16, 32)

    def __init__(self, model_path, score_th=0.5, use_cuda=True):
        self.session, self.provider = _create_ort_session(model_path, use_cuda)
        self.device = 'GPU' if self.provider == 'CUDAExecutionProvider' else 'CPU'
        self.score_th = score_th
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._centers = {}

    def with_threshold(self, score_th):
        """Salinan yang berbagi sesi ONNX Runtime, mis. untuk enrollment dengan ambang lebih rendah."""
        clone = copy.copy(self)
        clone.score_th = score_th
        return clone

    def setInputSize(self, size):
        """Tidak dipakai: ukuran input mengikuti gambar. Ada agar kompatibel dengan cv2.FaceDetectorYN."""

    def _anchor_centers(self, fh, fw, stride):
        key = (fh, fw, stride)
        if key not in self._centers:
            c = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
            c = (c * stride).reshape(-1, 2)
            self._centers[key] = np.stack([c, c], axis=1).reshape(-1, 2)  # 2 anchor per titik
        return self._centers[key]

    def detect(self, img):
        h, w = img.shape[:2]
        W, H = max(32, (w // 32) * 32), max(32, (h // 32) * 32)
        if float(h) / w > float(H) / W:
            nh, nw = H, int(H * w / float(h))
        else:
            nw, nh = W, int(W * h / float(w))
        scale = nh / float(h)
        det_img = np.zeros((H, W, 3), dtype=np.uint8)
        det_img[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128, (W, H), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})

        boxes, scores, kpss = [], [], []
        for i, stride in enumerate(self.strides):
            sc = outs[i].reshape(-1)
            keep = np.where(sc >= self.score_th)[0]
            if keep.size == 0:
                continue
            ctr = self._anchor_centers(H // stride, W // stride, stride)[keep]
            d = outs[i + 3].reshape(-1, 4)[keep] * stride
            k = outs[i + 6].reshape(-1, 10)[keep] * stride
            boxes.append(np.stack([ctr[:, 0] - d[:, 0], ctr[:, 1] - d[:, 1],
                                   ctr[:, 0] + d[:, 2], ctr[:, 1] + d[:, 3]], axis=1))
            kp = np.empty_like(k)
            kp[:, 0::2] = k[:, 0::2] + ctr[:, 0:1]
            kp[:, 1::2] = k[:, 1::2] + ctr[:, 1:2]
            kpss.append(kp)
            scores.append(sc[keep])
        if not scores:
            return 1, None

        boxes = np.concatenate(boxes) / scale
        kpss = np.concatenate(kpss) / scale
        scores = np.concatenate(scores)
        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        keep = np.array(cv2.dnn.NMSBoxes(xywh, [float(s) for s in scores], self.score_th, self.nms_th),
                        dtype=np.int64).flatten()

        rows = []
        for j in keep:
            # Bbox dipotong ke batas gambar, sama seperti make_face() di combo_pipeline.py
            x1, y1 = max(0.0, float(boxes[j, 0])), max(0.0, float(boxes[j, 1]))
            x2, y2 = min(float(w), float(boxes[j, 2])), min(float(h), float(boxes[j, 3]))
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append([x1, y1, x2 - x1, y2 - y1] + [float(v) for v in kpss[j]] + [float(scores[j])])
        if not rows:
            return 1, None
        return 1, np.array(rows, dtype=np.float32)


def load_scrfd_detector(use_cuda=True, score_th=0.5):
    """
    Muat SCRFD untuk skrip presensi. Mengembalikan None (dengan peringatan) jika onnxruntime atau
    model tidak tersedia, agar skrip tetap berjalan dengan YuNet.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("[WARN] onnxruntime belum terpasang: detector wajah kembali ke YuNet. "
              "Pasang `pip install onnxruntime` (Jetson: wheel onnxruntime-gpu dari NVIDIA).")
        return None
    try:
        detector = SCRFDFaceDetector(ensure_scrfd_model(), score_th=score_th, use_cuda=use_cuda)
    except Exception as e:
        print(f"[WARN] SCRFD gagal dimuat ({e}): detector wajah kembali ke YuNet.")
        return None
    print(f"[MODEL] Detector wajah: SCRFD-500M via ONNX Runtime ({detector.provider})")
    return detector
