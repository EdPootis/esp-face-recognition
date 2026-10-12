"""
===========================================================================
  BENCHMARK COMPARATOR - Analisis Perbandingan Hasil Benchmark
===========================================================================
Script ini membaca semua file JSON hasil benchmark di folder results/
dan menghasilkan tabel perbandingan komprehensif.

Cara Menjalankan:
  python compare_results.py
  python compare_results.py --results-dir ./results

Output:
  - Tabel perbandingan di terminal
  - File ringkasan TXT & JSON di folder results/
===========================================================================
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_utils import GUIDED_SCENARIOS, BENCHMARK_LOG_DIR


def load_all_results(results_dir):
    """Muat semua file JSON benchmark dari direktori."""
    results = []
    if not os.path.isdir(results_dir):
        print(f"  ❌ Folder hasil '{results_dir}' tidak ditemukan.")
        return results

    json_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.json')
                         and f.startswith('benchmark_')])

    for jf in json_files:
        path = os.path.join(results_dir, jf)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data["_source_file"] = jf
                results.append(data)
        except Exception as e:
            print(f"  ⚠️  Gagal membaca {jf}: {e}")

    return results


def print_comparison_table(results):
    """Cetak tabel perbandingan antar pasangan model."""
    if not results:
        print("  Tidak ada data benchmark untuk dibandingkan.")
        return

    w = 90

    print(f"\n{'=' * w}")
    print(f"{'PERBANDINGAN PASANGAN MODEL FD + FR':^{w}}")
    print(f"{'=' * w}")
    print(f"  Jumlah hasil benchmark: {len(results)}")
    print(f"  Tanggal analisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * w}")

    # ====== TABEL 1: OVERVIEW ======
    print(f"\n{'─' * w}")
    print(f"  TABEL 1: RINGKASAN UMUM")
    print(f"{'─' * w}")

    header = f"  {'No':>3} | {'Pasangan Model':<45} | {'Backend':<20} | {'File'}"
    print(header)
    print(f"  {'─' * 3}-+-{'─' * 45}-+-{'─' * 20}-+-{'─' * 15}")

    for i, r in enumerate(results, 1):
        pair = r.get("pair_name", "?")[:45]
        backend = r.get("model_info", {}).get("backend", "?")[:20]
        src = r.get("_source_file", "?")[:15]
        print(f"  {i:3d} | {pair:<45} | {backend:<20} | {src}")

    # ====== TABEL 2: LATENSI PER SKENARIO ======
    scenario_ids = [s["id"] for s in GUIDED_SCENARIOS]

    for sid in scenario_ids:
        sc_name = sid
        for gs in GUIDED_SCENARIOS:
            if gs["id"] == sid:
                sc_name = gs["name"]
                break

        print(f"\n{'─' * w}")
        print(f"  TABEL: SKENARIO '{sc_name}'")
        print(f"{'─' * w}")

        header = (f"  {'Pasangan Model':<35} | "
                  f"{'Det(ms)':>8} | {'Rec(ms)':>8} | {'Total(ms)':>9} | "
                  f"{'DetRate':>7} | {'RecRate':>7} | "
                  f"{'DetConf':>7} | {'RecScore':>8} | {'eFPS':>5}")
        print(header)
        print(f"  {'─' * 35}-+-{'─' * 8}-+-{'─' * 8}-+-{'─' * 9}-+-"
              f"{'─' * 7}-+-{'─' * 7}-+-{'─' * 7}-+-{'─' * 8}-+-{'─' * 5}")

        for r in results:
            pair = r.get("pair_name", "?")[:35]
            sc = r.get("scenarios", {}).get(sid)
            if sc is None:
                print(f"  {pair:<35} | {'N/A':>8} | {'N/A':>8} | {'N/A':>9} | "
                      f"{'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {'N/A':>8} | {'N/A':>5}")
                continue

            det_lat = sc.get("detection_latency", {}).get("mean", 0)
            rec_lat = sc.get("recognition_latency", {}).get("mean", 0)
            tot_lat = sc.get("total_pipeline_latency", {}).get("mean", 0)
            det_rate = sc.get("detection_rate", 0)
            rec_rate = sc.get("recognition_rate", 0)
            det_conf = sc.get("detection_confidence", {}).get("mean", 0)
            rec_score = sc.get("recognition_score", {}).get("mean", 0)
            efps = sc.get("effective_fps", 0)

            print(f"  {pair:<35} | {det_lat:8.1f} | {rec_lat:8.1f} | {tot_lat:9.1f} | "
                  f"{det_rate:6.1f}% | {rec_rate:6.1f}% | "
                  f"{det_conf:7.3f} | {rec_score:8.3f} | {efps:5.1f}")

    # ====== TABEL 3: RATA-RATA KESELURUHAN ======
    print(f"\n{'─' * w}")
    print(f"  TABEL: RATA-RATA KESELURUHAN (Semua Skenario)")
    print(f"{'─' * w}")

    header = (f"  {'Pasangan Model':<35} | "
              f"{'Avg Det(ms)':>11} | {'Avg Rec(ms)':>11} | {'Avg Tot(ms)':>11} | "
              f"{'Avg DetRate':>10} | {'Avg RecRate':>10}")
    print(header)
    print(f"  {'─' * 35}-+-{'─' * 11}-+-{'─' * 11}-+-{'─' * 11}-+-{'─' * 10}-+-{'─' * 10}")

    for r in results:
        pair = r.get("pair_name", "?")[:35]
        scenarios = r.get("scenarios", {})
        if not scenarios:
            print(f"  {pair:<35} | {'N/A':>11} | {'N/A':>11} | {'N/A':>11} | {'N/A':>10} | {'N/A':>10}")
            continue

        all_det = []
        all_rec = []
        all_tot = []
        all_det_rate = []
        all_rec_rate = []

        for sid, sc in scenarios.items():
            all_det.append(sc.get("detection_latency", {}).get("mean", 0))
            all_rec.append(sc.get("recognition_latency", {}).get("mean", 0))
            all_tot.append(sc.get("total_pipeline_latency", {}).get("mean", 0))
            all_det_rate.append(sc.get("detection_rate", 0))
            all_rec_rate.append(sc.get("recognition_rate", 0))

        avg_det = np.mean(all_det)
        avg_rec = np.mean(all_rec)
        avg_tot = np.mean(all_tot)
        avg_det_r = np.mean(all_det_rate)
        avg_rec_r = np.mean(all_rec_rate)

        print(f"  {pair:<35} | {avg_det:11.1f} | {avg_rec:11.1f} | {avg_tot:11.1f} | "
              f"{avg_det_r:9.1f}% | {avg_rec_r:9.1f}%")

    # ====== TABEL 4: RANKING ======
    print(f"\n{'─' * w}")
    print(f"  RANKING PASANGAN MODEL")
    print(f"{'─' * w}")

    rankings = []
    for r in results:
        pair = r.get("pair_name", "?")
        scenarios = r.get("scenarios", {})
        if not scenarios:
            continue

        all_tot = [sc.get("total_pipeline_latency", {}).get("mean", 999)
                   for sc in scenarios.values()]
        all_det_r = [sc.get("detection_rate", 0) for sc in scenarios.values()]
        all_rec_r = [sc.get("recognition_rate", 0) for sc in scenarios.values()]

        avg_lat = np.mean(all_tot)
        avg_det_r = np.mean(all_det_r)
        avg_rec_r = np.mean(all_rec_r)

        # Skor komposit: bobot 40% kecepatan, 30% det rate, 30% rec rate
        # Normalisasi kecepatan: semakin rendah latensi semakin baik
        speed_score = max(0, 100 - avg_lat)  # ms -> skor (100ms = 0, 0ms = 100)
        composite = speed_score * 0.4 + avg_det_r * 0.3 + avg_rec_r * 0.3

        rankings.append({
            "pair": pair,
            "avg_latency": avg_lat,
            "avg_det_rate": avg_det_r,
            "avg_rec_rate": avg_rec_r,
            "composite": composite,
        })

    rankings.sort(key=lambda x: -x["composite"])

    print(f"  {'Rank':>4} | {'Pasangan Model':<40} | {'Latency(ms)':>11} | "
          f"{'Det Rate':>8} | {'Rec Rate':>8} | {'Skor':>6}")
    print(f"  {'─' * 4}-+-{'─' * 40}-+-{'─' * 11}-+-{'─' * 8}-+-{'─' * 8}-+-{'─' * 6}")

    for rank, r in enumerate(rankings, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"  {medal}{rank:2d} | {r['pair']:<40} | {r['avg_latency']:11.1f} | "
              f"{r['avg_det_rate']:7.1f}% | {r['avg_rec_rate']:7.1f}% | {r['composite']:6.1f}")

    # ====== KESIMPULAN ======
    if rankings:
        best = rankings[0]
        print(f"\n{'─' * w}")
        print(f"  🏆 REKOMENDASI TERBAIK: {best['pair']}")
        print(f"     Latensi rata-rata: {best['avg_latency']:.1f}ms")
        print(f"     Detection rate: {best['avg_det_rate']:.1f}%")
        print(f"     Recognition rate: {best['avg_rec_rate']:.1f}%")
        print(f"{'─' * w}")


def save_comparison(results, results_dir):
    """Simpan ringkasan perbandingan ke file."""
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"comparison_{ts}.json")

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "num_benchmarks": len(results),
        "benchmarks": []
    }

    for r in results:
        entry = {
            "pair_name": r.get("pair_name"),
            "source_file": r.get("_source_file"),
            "model_info": r.get("model_info"),
            "device_info": r.get("device_info"),
            "scenarios": r.get("scenarios"),
        }
        comparison["benchmarks"].append(entry)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\n  💾 Perbandingan disimpan: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analisis & Bandingkan Hasil Benchmark Face AI Models"
    )
    parser.add_argument(
        '--results-dir', type=str, default=BENCHMARK_LOG_DIR,
        help=f"Direktori file JSON hasil benchmark (Default: {BENCHMARK_LOG_DIR})"
    )
    args = parser.parse_args()

    print(f"\n{'=' * 72}")
    print(f"{'BENCHMARK COMPARATOR':^72}")
    print(f"{'Analisis Perbandingan Pasangan Model FD + FR':^72}")
    print(f"{'=' * 72}")

    results = load_all_results(args.results_dir)

    if not results:
        print(f"\n  ⚠️  Tidak ada file benchmark ditemukan di: {args.results_dir}")
        print(f"  Jalankan minimal satu script benchmark terlebih dahulu:")
        print(f"    python benchmark_01_yunet_sface.py --stream-url <URL>")
        print(f"    python benchmark_02_blazeface_mobilefacenet.py --stream-url <URL>")
        print(f"    python benchmark_03_retinaface_arcface.py --stream-url <URL>")
        return

    print(f"\n  ✓ Ditemukan {len(results)} file benchmark.")
    for r in results:
        print(f"    • {r.get('pair_name', '?')} ({r.get('_source_file', '?')})")

    print_comparison_table(results)
    save_comparison(results, args.results_dir)

    print(f"\n{'=' * 72}")
    print(f"  ✅ ANALISIS PERBANDINGAN SELESAI")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
