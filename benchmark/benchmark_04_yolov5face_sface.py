"""
===========================================================================
  BENCHMARK #4: YOLOv5-Face + SFace
===========================================================================
Detector YOLO khusus wajah yang juga memprediksi 5 landmark, dipasangkan dengan
SFace (recognizer pasangan #1). Tujuan: memperbaiki kelemahan YuNet saat kepala
miring ke kiri (detection rate 20-42%) sambil mempertahankan keunggulan SFace di
jarak jauh dan cahaya redup.

Face Detection  : YOLOv5n-Face (1.7 M parameter) atau YOLOv5n-0.5-Face (0.45 M),
                  input 320x256, ONNX Runtime
Face Recognition: SFace (OpenCV alignCrop + OpenCV DNN / ONNX Runtime)

Dependensi: opencv-python, numpy, pillow, onnxruntime (Jetson: onnxruntime-gpu NVIDIA)
Model     : jalankan dulu benchmark/prepare_combo_models.py di laptop.

Cara Menjalankan:
  python benchmark_04_yolov5face_sface.py --video-file benchmark_video.mp4 --no-display
  python benchmark_04_yolov5face_sface.py --video-file benchmark_video.mp4 --yolo-model n-0.5   # Raspberry Pi 3
  python3 benchmark_04_yolov5face_sface.py --video-file benchmark_video.mp4 --rec-backend ort  # Jetson Nano

Lisensi: YOLOv5-Face GPL-3.0.
===========================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from combo_pipeline import (create_combo_argparser, run_pair_benchmark,
                            YOLOv5FaceDetector, SFaceRecognizer)

if __name__ == '__main__':
    parser = create_combo_argparser("YOLOv5-Face + SFace")
    parser.add_argument('--yolo-model', choices=['n', 'n-0.5'], default='n',
                        help="Varian YOLOv5-Face: 'n' (lebih akurat) atau 'n-0.5' (paling ringan, untuk Pi 3)\n"
                             "(Default: n)")
    parser.add_argument('--rec-backend', choices=['opencv', 'ort'], default='opencv',
                        help="Runtime SFace: 'opencv' (OpenCV DNN, sama dengan benchmark #1) atau\n"
                             "'ort' (ONNX Runtime, GPU jika ada; konfigurasi tercepat di Jetson). (Default: opencv)")
    args = parser.parse_args()

    use_cuda = not args.no_cuda
    variant_tag = args.yolo_model.replace('-', '').replace('.', '')  # n, n05
    pair_name = f"YOLOv5{args.yolo_model}-Face + SFace"
    pair_tag = f"yolov5{variant_tag}face_sface"

    print(f"\n{'=' * 72}\n  BENCHMARK: {pair_name}\n{'=' * 72}")
    detector = YOLOv5FaceDetector(args.yolo_model, use_cuda=use_cuda and not args.det_cpu, threads=args.threads)
    recognizer = SFaceRecognizer(args.rec_backend, use_cuda=use_cuda, threads=args.threads)
    run_pair_benchmark(pair_name, pair_tag, detector, recognizer, args)
