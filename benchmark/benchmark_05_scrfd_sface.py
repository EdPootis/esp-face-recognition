"""
===========================================================================
  BENCHMARK #5: SCRFD-500M + SFace  (tukar recognizer pasangan #3)
===========================================================================
Detector pasangan #3 (SCRFD-500M, input 320x224 sama persis) dipasangkan dengan
recognizer pasangan #1 (SFace). Dibandingkan dengan:
  - #3 (SCRFD + ArcFace-MBF): beda hanya di recognizer -> efek recognizer.
  - #1 (YuNet + SFace)       : beda hanya di detector   -> efek detector.
Pertanyaan utama: apakah kelemahan #3 di jarak jauh (recognition 14-17% padahal
detection 88-95%) berasal dari recognizer ArcFace-MBF?

Face Detection  : SCRFD-500M (InsightFace buffalo_sc det_500m), ONNX Runtime, tanpa InsightFace
Face Recognition: SFace (OpenCV alignCrop + OpenCV DNN / ONNX Runtime)

Dependensi: opencv-python, numpy, pillow, onnxruntime (Jetson: onnxruntime-gpu NVIDIA)
Model     : jalankan dulu benchmark/prepare_combo_models.py di laptop.

Cara Menjalankan:
  python benchmark_05_scrfd_sface.py --video-file benchmark_video.mp4 --no-display
  python3 benchmark_05_scrfd_sface.py --video-file benchmark_video.mp4 --rec-backend ort   # Jetson Nano

Lisensi: model InsightFace hanya untuk riset non-komersial.
===========================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from combo_pipeline import (create_combo_argparser, run_pair_benchmark,
                            SCRFDDetector, SFaceRecognizer)

PAIR_NAME = "SCRFD-500M + SFace"
PAIR_TAG = "scrfd_sface"

if __name__ == '__main__':
    parser = create_combo_argparser(PAIR_NAME)
    parser.add_argument('--rec-backend', choices=['opencv', 'ort'], default='opencv',
                        help="Runtime SFace: 'opencv' (OpenCV DNN, sama dengan benchmark #1) atau\n"
                             "'ort' (ONNX Runtime, GPU jika ada; konfigurasi tercepat di Jetson). (Default: opencv)")
    args = parser.parse_args()

    use_cuda = not args.no_cuda
    print(f"\n{'=' * 72}\n  BENCHMARK: {PAIR_NAME}\n{'=' * 72}")
    detector = SCRFDDetector((args.det_width, args.det_height), use_cuda=use_cuda and not args.det_cpu,
                             threads=args.threads)
    recognizer = SFaceRecognizer(args.rec_backend, use_cuda=use_cuda, threads=args.threads)
    run_pair_benchmark(PAIR_NAME, PAIR_TAG, detector, recognizer, args)
