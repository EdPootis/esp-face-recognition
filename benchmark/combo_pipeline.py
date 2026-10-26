"""
==========================================================================
  COMBO PIPELINE - Komponen Bersama Benchmark Kombinasi FD + FR (#4 - #7)
==========================================================================
Benchmark #1 - #3 menguji pasangan model "bawaan" (satu ekosistem per pasangan).
Benchmark #4 - #7 menukar detector dan recognizer antar pasangan agar terlihat
seberapa besar pengaruh masing-masing komponen:

  #4  YOLOv5n-Face  + SFace          (detector YOLO dengan 5 landmark)
  #5  SCRFD-500M    + SFace          (detector pasangan #3, recognizer pasangan #1)
  #6  YuNet         + ArcFace-MBF    (detector pasangan #1, recognizer pasangan #2/#3)
  #7  BlazeFace full-range + ArcFace-MBF dengan alignment landmark

Prinsip desain (ringan untuk Raspberry Pi 3 dan Jetson Nano):
  - Tanpa InsightFace / MediaPipe / PyTorch. Cukup numpy, OpenCV, ONNX Runtime,
    dan (khusus #7) TFLite interpreter (ai-edge-litert / tflite_runtime).
  - Semua decode (anchor, grid, landmark) ditulis ulang di numpy dan sudah
    diverifikasi sama dengan implementasi aslinya.
  - Kompatibel Python 3.6 (Jetson Nano JetPack 4.x).
  - Model disiapkan sekali di laptop dengan prepare_combo_models.py, lalu folder
    models/ disalin ke perangkat.

Urutan 5 landmark (koordinat gambar): mata kiri-gambar, mata kanan-gambar, hidung,
mulut kiri-gambar, mulut kanan-gambar. Urutan ini sama untuk YuNet, SCRFD,
YOLOv5-Face, dan template ArcFace, sehingga landmark bisa langsung dipakai
cv2.FaceRecognizerSF.alignCrop() maupun alignment ArcFace.
"""

import os
import sys
import glob
import time
import importlib
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_utils import (
    CameraStreamer, VideoFileStreamer, MetricsCollector, GuidedScenarioRunner,
    GUIDED_SCENARIOS, KNOWN_FACES_DIR, create_base_argparser,
    save_report, get_device_info, format_device_string, get_model_file_size
)

# ==============================================================================
# PATH MODEL
# ==============================================================================
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BENCH_DIR)
MODELS_DIR = os.path.join(BENCH_DIR, 'models')

YUNET_PATH = os.path.join(ROOT_DIR, 'face_detection_yunet_2023mar.onnx')
YUNET_URL = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
SFACE_PATH = os.path.join(ROOT_DIR, 'face_recognition_sface_2021dec.onnx')
SFACE_ORT_PATH = os.path.join(ROOT_DIR, 'face_recognition_sface_2021dec_ort.onnx')
SFACE_URL = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx'

SCRFD_PATH = os.path.join(MODELS_DIR, 'scrfd_500m.onnx')
ARCFACE_MBF_PATH = os.path.join(MODELS_DIR, 'mobilefacenet.onnx')  # == InsightFace w600k_mbf.onnx
BLAZEFACE_FULL_PATH = os.path.join(MODELS_DIR, 'face_detection_full_range.tflite')
YOLO_FACE_PATHS = {
    'n': os.path.join(MODELS_DIR, 'yolov5n-face_320x256.onnx'),
    'n-0.5': os.path.join(MODELS_DIR, 'yolov5n-0.5-face_320x256.onnx'),
}

PREPARE_HINT = ("Jalankan dulu di laptop (butuh internet + paket `onnx`):\n"
                "      python benchmark/prepare_combo_models.py\n"
                "    lalu salin folder benchmark/models/ ke perangkat.")

# Template 5 titik ArcFace 112x112 (insightface/utils/face_align.py)
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)
# Template 4 titik untuk BlazeFace (mata, hidung, titik tengah mulut). BlazeFace tidak
# memberi sudut mulut, jadi titik mulut = rata-rata kedua sudut mulut template ArcFace.
ARCFACE_DST_4 = np.vstack([ARCFACE_DST[:3], ARCFACE_DST[3:5].mean(axis=0, keepdims=True)])


def require_file(path, what):
    if not os.path.exists(path):
        print(f"\n  ❌ Model {what} tidak ditemukan: {path}")
        print(f"    {PREPARE_HINT}")
        sys.exit(1)


def download_if_missing(path, url):
    if not os.path.exists(path):
        print(f"  📥 Mengunduh {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)


# ==============================================================================
# RUNTIME HELPERS
# ==============================================================================
def create_ort_session(model_path, use_cuda=True, threads=0):
    """ONNX Runtime session dengan CUDA EP jika tersedia. Mengembalikan (session, provider)."""
    import onnxruntime as ort
    providers = []
    if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if threads > 0:
        so.intra_op_num_threads = threads
    # Matikan spin-wait thread pool. Default-nya thread ORT terus berputar di semua core
    # setelah run(), sehingga tahap berikutnya (session lain / OpenCV / TFLite) berebut CPU.
    # Diukur di laptop: recognition setelah detector ORT 21.9 -> 6.5 ms. Dampaknya lebih
    # besar di Pi 3 (4 core lemah).
    try:
        so.add_session_config_entry('session.intra_op.allow_spinning', '0')
    except Exception:
        pass
    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    return sess, sess.get_providers()[0]


def short_provider(provider):
    return 'GPU' if provider == 'CUDAExecutionProvider' else 'CPU'


def load_tflite_interpreter_class():
    """Cari TFLite Interpreter: ai-edge-litert (Pi), tflite_runtime (Jetson), tensorflow (laptop)."""
    for module_name in ('ai_edge_litert.interpreter', 'tflite_runtime.interpreter'):
        try:
            return importlib.import_module(module_name).Interpreter, module_name.split('.')[0]
        except ImportError:
            continue
    try:
        import tensorflow as tf
        return tf.lite.Interpreter, 'tensorflow'
    except ImportError:
        return None, None


def nms(boxes_xywh, scores, score_th, iou_th):
    """NMS OpenCV; mengembalikan array indeks (kompatibel output lama/baru NMSBoxes)."""
    if len(scores) == 0:
        return np.array([], dtype=np.int64)
    idx = cv2.dnn.NMSBoxes([[float(v) for v in b] for b in boxes_xywh],
                           [float(s) for s in scores], score_th, iou_th)
    return np.array(idx, dtype=np.int64).flatten()


def make_face(x1, y1, x2, y2, score, kps, frame_w, frame_h):
    """Bentuk dict wajah standar dengan bbox dipotong ke batas frame."""
    x1 = max(0.0, float(x1))
    y1 = max(0.0, float(y1))
    x2 = min(float(frame_w), float(x2))
    y2 = min(float(frame_h), float(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
        "confidence": float(score),
        "kps": np.asarray(kps, dtype=np.float32),
    }


# ==============================================================================
# FACE DETECTORS
# ==============================================================================
class YuNetDetector:
    """
    YuNet via OpenCV DNN, dijalankan persis seperti benchmark #1: frame di-resize
    ke det_size (default 320x240) lalu koordinat diskalakan balik. Selalu di CPU:
    di Jetson Nano YuNet CPU 48 ms vs OpenCV CUDA FP16 123 ms.
    """
    name = "YuNet 2023mar (OpenCV DNN CPU)"
    model_path = YUNET_PATH
    score_th = 0.6

    def __init__(self, det_size=(320, 240)):
        download_if_missing(YUNET_PATH, YUNET_URL)
        self.det_size = det_size
        self.runtime = "OpenCV DNN (CPU)"
        self.input_desc = f"{det_size[0]}x{det_size[1]}"
        self.detector = cv2.FaceDetectorYN.create(YUNET_PATH, '', det_size, self.score_th, 0.3, 5000)
        self.enroll_detector = cv2.FaceDetectorYN.create(YUNET_PATH, '', (320, 320), 0.3, 0.3, 5000)

    @staticmethod
    def _to_faces(rows, sx, sy, w, h):
        faces = []
        if rows is None:
            return faces
        for r in rows:
            x, y, bw, bh = r[0] * sx, r[1] * sy, r[2] * sx, r[3] * sy
            kps = r[4:14].reshape(5, 2) * np.array([sx, sy], dtype=np.float32)
            f = make_face(x, y, x + bw, y + bh, r[14], kps, w, h)
            if f is not None:
                faces.append(f)
        return faces

    def detect(self, frame):
        h, w = frame.shape[:2]
        dw, dh = self.det_size
        _, rows = self.detector.detect(cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR))
        return self._to_faces(rows, w / dw, h / dh, w, h)

    def detect_enroll(self, img):
        # Sama seperti enrollment benchmark #1: resolusi penuh, threshold 0.3
        h, w = img.shape[:2]
        self.enroll_detector.setInputSize((w, h))
        _, rows = self.enroll_detector.detect(img)
        return self._to_faces(rows, 1.0, 1.0, w, h)


class SCRFDDetector:
    """
    SCRFD-500M (InsightFace buffalo_sc det_500m.onnx) via ONNX Runtime tanpa InsightFace.
    Pre/post-processing = insightface/model_zoo/scrfd.py: resize mempertahankan rasio
    (pojok kiri atas), normalisasi (x-127.5)/128, 3 stride x 2 anchor, NMS 0.4.
    """
    name = "SCRFD-500M (buffalo_sc det_500m)"
    model_path = SCRFD_PATH
    score_th = 0.5
    nms_th = 0.4
    strides = (8, 16, 32)

    def __init__(self, det_size=(320, 240), use_cuda=True, threads=0):
        require_file(SCRFD_PATH, 'SCRFD-500M')
        # InsightFace mensyaratkan kelipatan 32 (sama seperti benchmark #3: 320x240 -> 320x224)
        self.input_size = ((det_size[0] // 32) * 32, (det_size[1] // 32) * 32)
        self.session, provider = create_ort_session(SCRFD_PATH, use_cuda, threads)
        self.runtime = f"ONNX Runtime ({provider})"
        self.input_desc = f"{self.input_size[0]}x{self.input_size[1]}"
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._centers = {}

    def _anchor_centers(self, fh, fw, stride):
        key = (fh, fw, stride)
        if key not in self._centers:
            c = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32)
            c = (c * stride).reshape(-1, 2)
            self._centers[key] = np.stack([c, c], axis=1).reshape(-1, 2)  # 2 anchor per titik
        return self._centers[key]

    def detect(self, frame, score_th=None):
        th = self.score_th if score_th is None else score_th
        h, w = frame.shape[:2]
        W, H = self.input_size
        if float(h) / w > float(H) / W:
            nh, nw = H, int(H * w / float(h))
        else:
            nw, nh = W, int(W * h / float(w))
        scale = nh / float(h)
        det_img = np.zeros((H, W, 3), dtype=np.uint8)
        det_img[:nh, :nw] = cv2.resize(frame, (nw, nh))
        blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128, (W, H), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})

        boxes, scores, kpss = [], [], []
        for i, stride in enumerate(self.strides):
            sc = outs[i].reshape(-1)
            keep = np.where(sc >= th)[0]
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
            return []
        boxes = np.concatenate(boxes) / scale
        kpss = np.concatenate(kpss) / scale
        scores = np.concatenate(scores)
        xywh = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
        faces = []
        for j in nms(xywh, scores, th, self.nms_th):
            f = make_face(boxes[j, 0], boxes[j, 1], boxes[j, 2], boxes[j, 3], scores[j],
                          kpss[j].reshape(5, 2), w, h)
            if f is not None:
                faces.append(f)
        return faces

    def detect_enroll(self, img):
        return self.detect(img, score_th=0.3)


class YOLOv5FaceDetector:
    """
    YOLOv5-Face (deepcam-cn/yolov5-face, ONNX dari rilis clibdev) yang sudah dipotong
    sebelum layer Detect dan diubah ke input 320x256 + opset 13 oleh
    prepare_combo_models.py. Decode box + 5 landmark dilakukan di numpy:
      xy = (sigmoid*2 - 0.5 + grid) * stride,  wh = (sigmoid*2)^2 * anchor,
      landmark = raw * anchor + grid * stride,  skor = obj * cls.
    Hasil decode identik dengan output model asli (selisih < 0.0002 px).
    """
    score_th = 0.5
    nms_th = 0.5
    strides = (8, 16, 32)
    anchors = np.array([[4, 5, 8, 10, 13, 16],
                        [23, 29, 43, 55, 73, 105],
                        [146, 217, 231, 300, 335, 433]], dtype=np.float32).reshape(3, 3, 2)

    def __init__(self, variant='n', use_cuda=True, threads=0):
        self.model_path = YOLO_FACE_PATHS[variant]
        require_file(self.model_path, f'YOLOv5{variant}-Face')
        self.name = f"YOLOv5{variant}-Face (deepcam-cn, 320x256)"
        self.session, provider = create_ort_session(self.model_path, use_cuda, threads)
        self.runtime = f"ONNX Runtime ({provider})"
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.in_h, self.in_w = int(inp.shape[2]), int(inp.shape[3])
        self.input_desc = f"{self.in_w}x{self.in_h} (letterbox)"
        self._grids = {}

    def _grid(self, ny, nx):
        key = (ny, nx)
        if key not in self._grids:
            yv, xv = np.mgrid[:ny, :nx]
            self._grids[key] = np.stack((xv, yv), axis=-1).astype(np.float32)  # (ny, nx, 2)
        return self._grids[key]

    def detect(self, frame, score_th=None):
        th = self.score_th if score_th is None else score_th
        h, w = frame.shape[:2]
        r = min(self.in_w / float(w), self.in_h / float(h))
        nw, nh = int(round(w * r)), int(round(h * r))
        canvas = np.full((self.in_h, self.in_w, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255, (self.in_w, self.in_h), (0, 0, 0), swapRB=True)
        outs = self.session.run(None, {self.input_name: blob})

        # obj * cls >= th mensyaratkan obj >= th, jadi saring logit obj dulu (hemat CPU di Pi)
        obj_logit_th = np.log(th / (1.0 - th))
        boxes, scores, kpss = [], [], []
        for i, (raw, stride) in enumerate(zip(outs, self.strides)):
            ny, nx = raw.shape[2], raw.shape[3]
            p = raw.reshape(3, 16, ny, nx).transpose(0, 2, 3, 1)  # (anchor, ny, nx, 16)
            a_idx, gy, gx = np.where(p[..., 4] >= obj_logit_th)
            if a_idx.size == 0:
                continue
            c = p[a_idx, gy, gx]  # (k, 16)
            sig = 1.0 / (1.0 + np.exp(-c[:, [0, 1, 2, 3, 4, 15]]))
            score = sig[:, 4] * sig[:, 5]
            ok = score >= th
            if not np.any(ok):
                continue
            c, sig, score = c[ok], sig[ok], score[ok]
            grid = self._grid(ny, nx)[gy[ok], gx[ok]]
            anc = self.anchors[i][a_idx[ok]]
            xy = (sig[:, 0:2] * 2.0 - 0.5 + grid) * stride
            wh = (sig[:, 2:4] * 2.0) ** 2 * anc
            lm = c[:, 5:15].reshape(-1, 5, 2) * anc[:, None, :] + (grid * stride)[:, None, :]
            boxes.append(np.concatenate([xy - wh / 2.0, xy + wh / 2.0], axis=1))
            kpss.append(lm)
            scores.append(score)
        if not scores:
            return []
        boxes = np.concatenate(boxes) / r
        kpss = np.concatenate(kpss) / r
        scores = np.concatenate(scores)
        xywh = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
        faces = []
        for j in nms(xywh, scores, th, self.nms_th):
            f = make_face(boxes[j, 0], boxes[j, 1], boxes[j, 2], boxes[j, 3], scores[j], kpss[j], w, h)
            if f is not None:
                faces.append(f)
        return faces

    def detect_enroll(self, img):
        return self.detect(img, score_th=0.3)


class BlazeFaceFullRangeDetector:
    """
    BlazeFace full-range (MediaPipe face_detection_full_range.tflite, untuk wajah s.d. ±5 m)
    via TFLite interpreter, tanpa MediaPipe. Input 192x192 letterbox, 48x48 anchor
    (stride 4, 1 anchor per sel), sigmoid score, NMS 0.3 - sama seperti backend tflite
    benchmark #2 untuk model short-range.
    Landmark yang dipakai: mata kanan-subjek, mata kiri-subjek, hidung, tengah mulut
    (4 titik; telinga dibuang). BlazeFace tidak memberi sudut mulut.
    """
    name = "BlazeFace full-range (MediaPipe .tflite)"
    model_path = BLAZEFACE_FULL_PATH
    INPUT_SIZE = 192
    GRID = 48
    score_th = 0.5
    nms_th = 0.3

    def __init__(self, threads=4):
        require_file(BLAZEFACE_FULL_PATH, 'BlazeFace full-range')
        interpreter_cls, source = load_tflite_interpreter_class()
        if interpreter_cls is None:
            print("\n  ❌ TFLite interpreter tidak ditemukan. Pasang salah satu:")
            print("    pip install ai-edge-litert   (Raspberry Pi 64-bit, Python >= 3.9)")
            print("    tflite_runtime 2.5 (wheel Coral, Jetson Nano Python 3.6)")
            sys.exit(1)
        self.interpreter = interpreter_cls(model_path=BLAZEFACE_FULL_PATH, num_threads=threads)
        self.interpreter.allocate_tensors()
        self.runtime = f"TFLite Interpreter ({source})"
        self.input_desc = f"{self.INPUT_SIZE}x{self.INPUT_SIZE} (letterbox)"
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.boxes_index = self.scores_index = None
        for out in self.interpreter.get_output_details():
            if out['shape'][-1] == 16:
                self.boxes_index = out['index']
            elif out['shape'][-1] == 1:
                self.scores_index = out['index']
        g = (np.arange(self.GRID, dtype=np.float32) + 0.5) / self.GRID
        gx, gy = np.meshgrid(g, g)
        self.anchors = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)  # 2304 anchor

    def detect(self, frame, score_th=None):
        th = self.score_th if score_th is None else score_th
        h, w = frame.shape[:2]
        side = max(h, w)
        pad_x, pad_y = (side - w) // 2, (side - h) // 2
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
        keep = np.where(scores >= th)[0]
        if keep.size == 0:
            return []
        reg = raw_boxes[keep] / self.INPUT_SIZE
        anc = self.anchors[keep]
        scores = scores[keep]
        cx = (reg[:, 0] + anc[:, 0]) * side - pad_x
        cy = (reg[:, 1] + anc[:, 1]) * side - pad_y
        bw = reg[:, 2] * side
        bh = reg[:, 3] * side
        kx = (reg[:, 4::2] + anc[:, 0:1]) * side - pad_x
        ky = (reg[:, 5::2] + anc[:, 1:2]) * side - pad_y
        xywh = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1)
        faces = []
        for j in nms(xywh, scores, th, self.nms_th):
            kps = np.stack([kx[j, :4], ky[j, :4]], axis=1)
            f = make_face(cx[j] - bw[j] / 2, cy[j] - bh[j] / 2, cx[j] + bw[j] / 2, cy[j] + bh[j] / 2,
                          scores[j], kps, w, h)
            if f is not None:
                faces.append(f)
        return faces

    def detect_enroll(self, img):
        return self.detect(img, score_th=0.3)


# ==============================================================================
# FACE RECOGNIZERS (embedding ter-normalisasi L2, cocokkan dengan dot product)
# ==============================================================================
def umeyama_similarity(src, dst):
    """Transformasi similarity (rotasi + skala + translasi) least-squares src -> dst.
    Sama dengan skimage SimilarityTransform.estimate() yang dipakai InsightFace."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num, dim = src.shape
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_d, dst_d = src - src_mean, dst - dst_mean
    A = dst_d.T.dot(src_d) / num
    d = np.ones(dim)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    U, S, Vt = np.linalg.svd(A)
    R = U.dot(np.diag(d)).dot(Vt)
    scale = S.dot(d) / src_d.var(axis=0).sum()
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = R * scale
    M[:, 2] = dst_mean - scale * R.dot(src_mean)
    return M


class SFaceRecognizer:
    """
    SFace 128-d. alignCrop() OpenCV memakai baris berformat YuNet (bbox, 5 landmark,
    skor), sehingga landmark dari detector lain cukup dikemas ulang ke format itu.
    --rec-backend ort: inferensi via ONNX Runtime (konfigurasi tercepat di Jetson).
    """
    name = "SFace 2021dec"
    default_threshold = 0.363
    embedding_dim = 128
    model_path = SFACE_PATH

    def __init__(self, backend='opencv', use_cuda=True, threads=0):
        download_if_missing(SFACE_PATH, SFACE_URL)
        self.cv = cv2.FaceRecognizerSF.create(SFACE_PATH, '')
        self.session = None
        if backend == 'ort':
            path = SFACE_ORT_PATH if os.path.exists(SFACE_ORT_PATH) else SFACE_PATH
            self.session, provider = create_ort_session(path, use_cuda, threads)
            self.input_name = self.session.get_inputs()[0].name
            self.runtime = f"ONNX Runtime ({provider})"
            for _ in range(3):  # warm-up CUDA
                self._feature(np.zeros((112, 112, 3), dtype=np.uint8))
        else:
            self.runtime = "OpenCV DNN (CPU)"

    def _feature(self, aligned):
        if self.session is None:
            return self.cv.feature(aligned)
        blob = cv2.dnn.blobFromImage(aligned, 1.0, (112, 112), (0, 0, 0), swapRB=True, crop=False)
        return self.session.run(None, {self.input_name: blob})[0]

    def embed(self, frame, face):
        if len(face["kps"]) != 5:
            raise ValueError("SFace butuh 5 landmark")
        x, y, bw, bh = face["bbox"]
        row = np.zeros(15, dtype=np.float32)
        row[0:4] = (x, y, bw, bh)
        row[4:14] = face["kps"].reshape(-1)
        row[14] = face["confidence"]
        emb = self._feature(self.cv.alignCrop(frame, row)).reshape(-1).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-12)


class ArcFaceMBFRecognizer:
    """
    ArcFace MobileFaceNet 512-d (InsightFace w600k_mbf.onnx; file models/mobilefacenet.onnx
    identik byte-per-byte). Berbeda dari benchmark #2, wajah di-align ke template ArcFace
    112x112 memakai landmark (5 titik, atau 4 titik untuk BlazeFace), dan normalisasi
    (x-127.5)/127.5 sama dengan InsightFace.
    """
    name = "ArcFace MobileFaceNet (w600k_mbf)"
    default_threshold = 0.45  # sama dengan benchmark #3 untuk model yang sama
    embedding_dim = 512
    model_path = ARCFACE_MBF_PATH

    def __init__(self, use_cuda=True, threads=0):
        require_file(ARCFACE_MBF_PATH, 'ArcFace MobileFaceNet')
        self.session, provider = create_ort_session(ARCFACE_MBF_PATH, use_cuda, threads)
        self.runtime = f"ONNX Runtime ({provider})"
        self.input_name = self.session.get_inputs()[0].name
        for _ in range(3):
            self.session.run(None, {self.input_name: np.zeros((1, 3, 112, 112), dtype=np.float32)})

    def align(self, frame, face):
        kps = face["kps"]
        dst = ARCFACE_DST if len(kps) == 5 else ARCFACE_DST_4
        M = umeyama_similarity(kps, dst)
        return cv2.warpAffine(frame, M, (112, 112), borderValue=0.0)

    def embed(self, frame, face):
        aligned = self.align(frame, face)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        emb = self.session.run(None, {self.input_name: blob})[0].reshape(-1).astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-12)


# ==============================================================================
# ENROLLMENT & MATCHING
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
    names, embs = [], []
    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"  ⚠️  Folder '{KNOWN_FACES_DIR}' tidak ditemukan.")
        return names, np.zeros((0, recognizer.embedding_dim), dtype=np.float32)

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
            faces = detector.detect_enroll(img)
            if not faces:
                continue
            best = max(faces, key=lambda f: f["confidence"])
            names.append(person_name)
            embs.append(recognizer.embed(img, best))
            count += 1
        print(f"    ✓ '{person_name}': {count} embedding")

    print(f"  Total: {len(names)} embedding dari {len(set(names))} orang\n")
    if not embs:
        return names, np.zeros((0, recognizer.embedding_dim), dtype=np.float32)
    return names, np.vstack(embs)


def recognize_face(embedding, names, db, threshold):
    if len(names) == 0:
        return "Unknown", -1.0
    scores = db.dot(embedding)
    j = int(np.argmax(scores))
    best = float(scores[j])
    if best < threshold:
        return "Unknown", best
    return names[j], best


# ==============================================================================
# ARGUMENT PARSER & BENCHMARK LOOP
# ==============================================================================
def create_combo_argparser(pair_name):
    parser = create_base_argparser(pair_name)
    parser.add_argument(
        '--det-cpu', action='store_true',
        help="Jalankan detector di CPU walau CUDA tersedia (recognizer tetap di GPU).\n"
             "Model detector kecil bisa lebih cepat di CPU Jetson karena overhead transfer CPU<->GPU."
    )
    parser.add_argument(
        '--threads', type=int, default=0,
        help="Jumlah thread intra-op ONNX Runtime (Default: 0 = otomatis, semua core)"
    )
    return parser


def run_pair_benchmark(pair_name, pair_tag, detector, recognizer, args):
    """Loop benchmark guided, identik dengan benchmark #1 - #3 (5 skenario, metrik sama)."""
    device_info = get_device_info()
    device_str = format_device_string(device_info)
    print(f"\n  Device: {device_str}")
    print(f"  Detector  : {detector.name} | {detector.runtime} | input {detector.input_desc}")
    print(f"  Recognizer: {recognizer.name} | {recognizer.runtime}")

    threshold = args.threshold if args.threshold is not None else recognizer.default_threshold
    names, db = enroll_known_faces(detector, recognizer)

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
        print("\n  ❌ GAGAL mendapatkan frame dalam 15 detik!")
        streamer.stop()
        return

    model_info = {
        "detection_model": detector.name,
        "detection_model_size_bytes": get_model_file_size(detector.model_path),
        "recognition_model": recognizer.name,
        "recognition_model_size_bytes": get_model_file_size(recognizer.model_path),
        "backend": f"{detector.runtime} + {recognizer.runtime}",
        "det_input_size": detector.input_desc,
        "recognition_threshold": threshold,
        "embedding_dim": recognizer.embedding_dim,
        "alignment": "landmark (similarity transform)",
        "ort_threads": args.threads or "auto",
        "ort_allow_spinning": False,
    }

    runner = GuidedScenarioRunner()
    collector = MetricsCollector(pair_name)
    runner.print_header(pair_name, device_str)

    window_name = f"Benchmark: {pair_name}"
    if not args.no_display:
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
                faces = detector.detect(frame)
                det_latency = (time.perf_counter() - t0) * 1000

                face_detected = len(faces) > 0
                rec_latency = 0.0
                det_conf = 0.0
                rec_score = 0.0
                rec_name = "Unknown"
                face_recognized = False

                if face_detected:
                    best = max(faces, key=lambda f: f["confidence"])
                    det_conf = best["confidence"]
                    # === RECOGNITION (alignment + embedding + matching) ===
                    t2 = time.perf_counter()
                    emb = recognizer.embed(frame, best)
                    rec_name, rec_score = recognize_face(emb, names, db, threshold)
                    rec_latency = (time.perf_counter() - t2) * 1000
                    face_recognized = rec_name != "Unknown"

                total_latency = det_latency + rec_latency
                s = collector.scenarios[scenario["id"]]
                collector.record_frame(
                    det_latency_ms=det_latency, rec_latency_ms=rec_latency,
                    total_latency_ms=total_latency, face_detected=face_detected,
                    face_recognized=face_recognized, det_confidence=det_conf,
                    rec_score=rec_score, rec_name=rec_name, stream_fps=stream_fps
                )
                runner.print_progress(elapsed, args.duration, s["frames_processed"],
                                      s["faces_detected_count"], s["faces_recognized_count"], stream_fps)

                if not args.no_display:
                    disp = frame.copy()
                    for f in faces:
                        x, y, bw, bh = f["bbox"]
                        color = (0, 255, 0) if face_recognized else (0, 165, 255)
                        cv2.rectangle(disp, (x, y), (x + bw, y + bh), color, 2)
                        for px, py in f["kps"]:
                            cv2.circle(disp, (int(px), int(py)), 2, (255, 255, 0), -1)
                        label = f"{rec_name} ({rec_score:.2f})" if face_recognized else f"Det ({det_conf:.2f})"
                        cv2.putText(disp, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    cv2.putText(disp, f"{pair_tag} | {scenario['id']} | {total_latency:.0f}ms | FPS:{stream_fps:.1f}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow(window_name, disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\n\n  ⏹️  Benchmark dihentikan oleh pengguna.")
                        raise KeyboardInterrupt

                time.sleep(0.005)

            collector.end_scenario()
            runner.print_scenario_result(collector.get_scenario_summary(scenario["id"]))

    except KeyboardInterrupt:
        collector.end_scenario()
        print("\n  ⏹️  Benchmark dihentikan.")

    finally:
        # Simpan laporan SEBELUM streamer.stop(): di Jetson, FFmpeg bisa abort
        # (Assertion fctx->async_lock failed) saat cap.release() sehingga hasil hilang.
        report = collector.get_full_report()
        report["device_info"] = device_info
        report["model_info"] = model_info
        save_report(report, pair_tag)
        streamer.stop()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"\n{'=' * 72}")
    print(f"  ✅ BENCHMARK SELESAI: {pair_name}")
    print(f"{'=' * 72}")
