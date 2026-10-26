"""
===========================================================================
  PERSIAPAN MODEL untuk Benchmark Kombinasi #4 - #7
===========================================================================
Dijalankan SEKALI di laptop (butuh internet + paket `onnx`), lalu salin folder
benchmark/models/ (dan face_recognition_sface_2021dec_ort.onnx jika memakai
--rec-backend ort) ke Raspberry Pi / Jetson Nano. Perangkat edge tidak butuh
paket `onnx` maupun internet.

Yang dilakukan:
  1. SCRFD-500M + ArcFace MobileFaceNet : ambil det_500m.onnx & w600k_mbf.onnx dari
     paket InsightFace buffalo_sc (cache ~/.insightface atau unduh zip rilis).
  2. BlazeFace full-range                : unduh face_detection_full_range.tflite (MediaPipe).
  3. YOLOv5-Face (n dan n-0.5)           : unduh ONNX rilis clibdev/yolov5-face lalu konversi:
       - dipotong sebelum layer Detect (decode dikerjakan numpy di combo_pipeline.py),
       - konstanta Reshape (channel shuffle) diubah ke input 320x256 (default),
       - opset 18 -> 13 agar bisa dimuat ONNX Runtime 1.11 di Jetson Nano.
     Hasil konversi diverifikasi: pada input 640 decode numpy == output model asli.
  4. SFace untuk ONNX Runtime            : salinan SFace dengan initializer dihapus dari
     graph input (agar ONNX Runtime bisa constant folding).

Semua unduhan dicek SHA-256.

Cara Menjalankan:
  python benchmark/prepare_combo_models.py
  python benchmark/prepare_combo_models.py --yolo-size 320x320
===========================================================================
"""

import os
import sys
import shutil
import zipfile
import hashlib
import argparse
import tempfile
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from combo_pipeline import (MODELS_DIR, SCRFD_PATH, ARCFACE_MBF_PATH, BLAZEFACE_FULL_PATH,
                            SFACE_PATH, SFACE_ORT_PATH, SFACE_URL)

BUFFALO_SC_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip'
BLAZEFACE_FULL_URL = 'https://storage.googleapis.com/mediapipe-assets/face_detection_full_range.tflite'
YOLO_URLS = {
    'n': 'https://github.com/clibdev/yolov5-face/releases/download/1.0.0/yolov5n-face.onnx',
    'n-0.5': 'https://github.com/clibdev/yolov5-face/releases/download/1.0.0/yolov5n-0.5-face.onnx',
}
SHA256 = {
    'det_500m.onnx': '5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a',
    'w600k_mbf.onnx': '9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f',
    'face_detection_full_range.tflite': '3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b',
    'yolov5n-face.onnx': 'ee6ba4ccdc3c075d205c9703aec53a2aa3010c8d7fa08b0eff078e33a4b4fe6c',
    'yolov5n-0.5-face.onnx': '269eb1e54313f9d1f7941ed9939fa247767539bca5801fc7aa7895960e93ca43',
}


def rel(path):
    """Path relatif untuk ditampilkan (aman bila beda drive di Windows)."""
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(path, name):
    got = sha256(path)
    if got != SHA256[name]:
        raise RuntimeError(f"SHA-256 {name} tidak cocok ({got}). File rusak atau sudah berubah di sumbernya.")


def download(url, dest):
    print(f"  📥 {url}")
    urllib.request.urlretrieve(url, dest)


# ------------------------------------------------------------------------------
def prepare_buffalo_sc(tmp):
    """det_500m.onnx -> models/scrfd_500m.onnx, w600k_mbf.onnx -> models/mobilefacenet.onnx."""
    targets = {'det_500m.onnx': SCRFD_PATH, 'w600k_mbf.onnx': ARCFACE_MBF_PATH}
    missing = {k: v for k, v in targets.items() if not os.path.exists(v)}
    if not missing:
        print("  ✓ SCRFD-500M & ArcFace MobileFaceNet sudah ada.")
        return
    cache = os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'buffalo_sc')
    zip_path = None
    for name, dest in missing.items():
        src = os.path.join(cache, name)
        if not os.path.exists(src):
            if zip_path is None:
                zip_path = os.path.join(tmp, 'buffalo_sc.zip')
                download(BUFFALO_SC_URL, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                member = [m for m in z.namelist() if m.endswith(name)][0]
                src = os.path.join(tmp, name)
                with z.open(member) as fi, open(src, 'wb') as fo:
                    shutil.copyfileobj(fi, fo)
        verify(src, name)
        shutil.copyfile(src, dest)
        print(f"  ✓ {name} -> {rel(dest)}")


def prepare_blazeface_full():
    if os.path.exists(BLAZEFACE_FULL_PATH):
        print("  ✓ BlazeFace full-range sudah ada.")
        return
    download(BLAZEFACE_FULL_URL, BLAZEFACE_FULL_PATH)
    verify(BLAZEFACE_FULL_PATH, 'face_detection_full_range.tflite')
    print(f"  ✓ {rel(BLAZEFACE_FULL_PATH)}")


# ------------------------------------------------------------------------------
YOLO_ANCHORS = np.array([[4, 5, 8, 10, 13, 16],
                         [23, 29, 43, 55, 73, 105],
                         [146, 217, 231, 300, 335, 433]], dtype=np.float32).reshape(3, 3, 2)


def yolo_decode_all(outs):
    """Decode penuh (semua anchor) - hanya untuk verifikasi terhadap output model asli."""
    rows = []
    for i, (raw, s) in enumerate(zip(outs, (8, 16, 32))):
        _, _, ny, nx = raw.shape
        p = raw.reshape(3, 16, ny, nx).transpose(0, 2, 3, 1)
        yv, xv = np.mgrid[:ny, :nx]
        grid = np.stack((xv, yv), -1)[None].astype(np.float32)
        a = YOLO_ANCHORS[i][:, None, None, :]
        sig = 1.0 / (1.0 + np.exp(-p))
        xy = (sig[..., 0:2] * 2 - 0.5 + grid) * s
        wh = (sig[..., 2:4] * 2) ** 2 * a
        lm = p[..., 5:15] * np.tile(a, (1, 1, 1, 5)) + np.tile(grid * s, (1, 1, 1, 5))
        rows.append(np.concatenate([xy, wh, sig[..., 4:5], lm, sig[..., 15:16]], -1).reshape(-1, 16))
    return np.concatenate(rows)


def convert_yolo(src, dst, width, height):
    import onnx
    from onnx import numpy_helper, helper

    def build(out_path, W, H):
        heads = ['/model.21/m.%d/Conv_output_0' % i for i in range(3)]
        onnx.utils.extract_model(src, out_path, ['input'], heads)
        m = onnx.load(out_path)
        consts = {n.output[0]: n for n in m.graph.node if n.op_type == 'Constant'}
        # Ukuran spasial channel-shuffle tertanam di konstanta Reshape (80/40/20 untuk input 640)
        hmap = {80: H // 8, 40: H // 16, 20: H // 32}
        wmap = {80: W // 8, 40: W // 16, 20: W // 32}
        for n in m.graph.node:
            if n.op_type != 'Reshape':
                continue
            c = consts[n.input[1]]
            arr = numpy_helper.to_array(c.attribute[0].t).copy()
            arr[-2], arr[-1] = hmap[int(arr[-2])], wmap[int(arr[-1])]
            c.attribute[0].t.CopyFrom(numpy_helper.from_array(arr.astype(np.int64), c.attribute[0].t.name))
            for a in list(n.attribute):
                if a.name == 'allowzero':  # atribut opset 14, tidak ada di opset 13
                    n.attribute.remove(a)
        dims = m.graph.input[0].type.tensor_type.shape.dim
        dims[2].dim_value, dims[3].dim_value = H, W
        for o, st in zip(m.graph.output, (8, 16, 32)):
            od = o.type.tensor_type.shape.dim
            del od[:]
            for v in (1, 48, H // st, W // st):
                od.add().dim_value = v
        del m.graph.value_info[:]
        del m.opset_import[:]
        m.opset_import.extend([helper.make_opsetid('', 13)])
        m.ir_version = 7
        onnx.checker.check_model(m, full_check=True)
        onnx.save(m, out_path)

    import onnxruntime as ort
    # Verifikasi: model terpotong @640 + decode numpy == output model asli
    check_path = dst + '.check640.onnx'
    build(check_path, 640, 640)
    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    ref = ort.InferenceSession(src, providers=['CPUExecutionProvider']).run(None, {'input': x})[0][0]
    mine = yolo_decode_all(ort.InferenceSession(check_path, providers=['CPUExecutionProvider']).run(None, {'input': x}))
    os.remove(check_path)
    diff = float(np.abs(ref - mine).max())
    if diff > 0.01:
        raise RuntimeError(f"Verifikasi decode YOLO gagal (max|diff| = {diff})")

    build(dst, width, height)
    sess = ort.InferenceSession(dst, providers=['CPUExecutionProvider'])
    outs = sess.run(None, {'input': np.zeros((1, 3, height, width), dtype=np.float32)})
    print(f"  ✓ {rel(dst)}  (input {width}x{height}, opset 13, output "
          f"{[o.shape for o in outs]}, decode vs asli @640 max|diff| {diff:.1e} px)")


def prepare_yolo(tmp, width, height, force=False):
    for variant, url in YOLO_URLS.items():
        name = os.path.basename(url)
        dst = os.path.join(MODELS_DIR, name.replace('.onnx', f'_{width}x{height}.onnx'))
        if os.path.exists(dst) and not force:
            print(f"  ✓ {rel(dst)} sudah ada.")
            continue
        src = os.path.join(tmp, name)
        download(url, src)
        verify(src, name)
        convert_yolo(src, dst, width, height)


def prepare_sface_ort():
    if os.path.exists(SFACE_ORT_PATH):
        print(f"  ✓ {os.path.basename(SFACE_ORT_PATH)} sudah ada.")
        return
    import onnx
    if not os.path.exists(SFACE_PATH):
        download(SFACE_URL, SFACE_PATH)
    m = onnx.load(SFACE_PATH)
    init_names = set(i.name for i in m.graph.initializer)
    keep = [i for i in m.graph.input if i.name not in init_names]
    del m.graph.input[:]
    m.graph.input.extend(keep)
    onnx.save(m, SFACE_ORT_PATH)
    print(f"  ✓ {os.path.basename(SFACE_ORT_PATH)} ({len(init_names)} initializer dihapus dari graph input)")


def main():
    parser = argparse.ArgumentParser(description="Siapkan model untuk benchmark kombinasi #4 - #7")
    parser.add_argument('--yolo-size', default='320x256',
                        help="Ukuran input YOLOv5-Face WxH, kelipatan 32 (Default: 320x256, pas untuk frame 4:3)")
    parser.add_argument('--force-yolo', action='store_true', help="Konversi ulang YOLOv5-Face walau file sudah ada")
    args = parser.parse_args()
    width, height = [int(v) for v in args.yolo_size.lower().split('x')]
    if width % 32 or height % 32:
        parser.error("--yolo-size harus kelipatan 32, mis. 320x256 atau 320x320")
    if (width, height) != (320, 256):
        print("  ℹ️  combo_pipeline.py memuat file *_320x256.onnx; ganti YOLO_FACE_PATHS bila memakai ukuran lain.")

    try:
        import onnx  # noqa: F401
    except ImportError:
        print("  ❌ Paket `onnx` dibutuhkan untuk konversi (pip install onnx). Jalankan di laptop.")
        sys.exit(1)

    os.makedirs(MODELS_DIR, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='combo_models_')
    try:
        print("\n[1/4] SCRFD-500M + ArcFace MobileFaceNet (InsightFace buffalo_sc)")
        prepare_buffalo_sc(tmp)
        print("\n[2/4] BlazeFace full-range")
        prepare_blazeface_full()
        print("\n[3/4] YOLOv5-Face")
        prepare_yolo(tmp, width, height, force=args.force_yolo)
        print("\n[4/4] SFace untuk ONNX Runtime")
        prepare_sface_ort()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n✅ Selesai. Salin folder {rel(MODELS_DIR)}/ (dan "
          f"{os.path.basename(SFACE_ORT_PATH)} untuk --rec-backend ort) ke Raspberry Pi / Jetson Nano.")
    print("   Lisensi: YOLOv5-Face GPL-3.0; model InsightFace (SCRFD, ArcFace) hanya untuk riset non-komersial;")
    print("   YuNet MIT, SFace Apache-2.0, BlazeFace Apache-2.0.")


if __name__ == '__main__':
    main()
