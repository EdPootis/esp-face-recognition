"""
===========================================================================
  BENCHMARK #6: YuNet + ArcFace-MobileFaceNet  (tukar recognizer pasangan #1)
===========================================================================
Detector pasangan #1 (YuNet, 320x240, alur sama persis dengan benchmark #1)
dipasangkan dengan recognizer pasangan #2/#3 (ArcFace MobileFaceNet w600k_mbf)
yang di-align dengan 5 landmark. Pasangan cermin dari #5:
  - #1 (YuNet + SFace)  : beda hanya di recognizer -> efek recognizer.
  - #3 (SCRFD + ArcFace): beda hanya di detector   -> efek detector.

Face Detection  : YuNet 2023mar (OpenCV DNN, CPU)
Face Recognition: ArcFace MobileFaceNet 512-d (ONNX Runtime), alignment 5 titik
                  seperti InsightFace, threshold 0.45 (sama dengan benchmark #3)

Dependensi: opencv-python, numpy, pillow, onnxruntime (Jetson: onnxruntime-gpu NVIDIA)
Model     : jalankan dulu benchmark/prepare_combo_models.py di laptop.

Cara Menjalankan:
  python benchmark_06_yunet_arcface.py --video-file benchmark_video.mp4 --no-display

Lisensi: model InsightFace hanya untuk riset non-komersial.
===========================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from combo_pipeline import (create_combo_argparser, run_pair_benchmark,
                            YuNetDetector, ArcFaceMBFRecognizer)

PAIR_NAME = "YuNet + ArcFace-MobileFaceNet"
PAIR_TAG = "yunet_arcface"

if __name__ == '__main__':
    parser = create_combo_argparser(PAIR_NAME)
    args = parser.parse_args()

    print(f"\n{'=' * 72}\n  BENCHMARK: {PAIR_NAME}\n{'=' * 72}")
    detector = YuNetDetector((args.det_width, args.det_height))  # selalu CPU (lihat docstring)
    recognizer = ArcFaceMBFRecognizer(use_cuda=not args.no_cuda, threads=args.threads)
    run_pair_benchmark(PAIR_NAME, PAIR_TAG, detector, recognizer, args)
