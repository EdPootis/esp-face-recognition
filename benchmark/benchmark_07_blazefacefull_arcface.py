"""
===========================================================================
  BENCHMARK #7: BlazeFace Full-Range + ArcFace-MobileFaceNet (dengan alignment)
===========================================================================
Uji ulang pasangan #2 secara adil. Dua kekurangan #2 diperbaiki:
  1. Model short-range (dirancang untuk wajah < 2 m / selfie) diganti full-range
     (wajah s.d. ±5 m) -> relevan untuk skenario jarak jauh.
  2. Wajah di-align ke template ArcFace memakai landmark BlazeFace (mata, hidung,
     tengah mulut) sebelum embedding, bukan sekadar crop bounding box.
Recognizer-nya identik dengan #2/#3 (w600k_mbf), threshold 0.45 seperti #3.
Catatan: #2 memakai threshold 0.55 untuk model yang sama; jalankan dengan
--threshold 0.55 bila ingin membandingkan dengan #2 pada threshold yang sama.

Face Detection  : BlazeFace full-range (MediaPipe .tflite, 192x192), TFLite interpreter
                  tanpa MediaPipe (aman untuk CPU Pi 3 yang tidak punya instruksi AES)
Face Recognition: ArcFace MobileFaceNet 512-d (ONNX Runtime), alignment 4 titik

Dependensi: opencv-python, numpy, pillow, onnxruntime, dan TFLite interpreter:
  Raspberry Pi: pip install ai-edge-litert
  Jetson Nano : tflite_runtime 2.5 (wheel Coral Python 3.6)
Model     : jalankan dulu benchmark/prepare_combo_models.py di laptop.

Cara Menjalankan:
  python benchmark_07_blazefacefull_arcface.py --video-file benchmark_video.mp4 --no-display
===========================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from combo_pipeline import (create_combo_argparser, run_pair_benchmark,
                            BlazeFaceFullRangeDetector, ArcFaceMBFRecognizer)

PAIR_NAME = "BlazeFace Full-Range + ArcFace-MobileFaceNet (aligned)"
PAIR_TAG = "blazefacefull_arcface"

if __name__ == '__main__':
    parser = create_combo_argparser(PAIR_NAME)
    parser.add_argument('--tflite-threads', type=int, default=4,
                        help="Jumlah thread TFLite interpreter (Default: 4 = semua core Pi 3)")
    args = parser.parse_args()

    print(f"\n{'=' * 72}\n  BENCHMARK: {PAIR_NAME}\n{'=' * 72}")
    detector = BlazeFaceFullRangeDetector(threads=args.tflite_threads)
    recognizer = ArcFaceMBFRecognizer(use_cuda=not args.no_cuda, threads=args.threads)
    run_pair_benchmark(PAIR_NAME, PAIR_TAG, detector, recognizer, args)
