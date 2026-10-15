"""
Membuat grafik perbandingan 2x2 (detection rate, recognition rate, latency, FPS)
dari file comparison_*.json hasil compare_results.py.

Cara Menjalankan:
  python make_graph.py                         # pakai comparison_*.json terbaru di results/ (termasuk subfolder)
  python make_graph.py results/comparison_20261015_085225.json
  python make_graph.py --no-show               # hanya simpan PNG (mis. via SSH)
"""

import os
import re
import glob
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def find_latest_comparison(results_dir):
    """Cari comparison_*.json terbaru (berdasarkan timestamp di nama file) di results_dir dan subfoldernya."""
    files = glob.glob(os.path.join(results_dir, '**', 'comparison_*.json'), recursive=True)
    if not files:
        return None

    def sort_key(path):
        m = re.search(r'comparison_(\d{8}_\d{6})', os.path.basename(path))
        return (m.group(1) if m else '', os.path.getmtime(path))

    return max(files, key=sort_key)


def build_model_entries(benchmarks):
    """
    Deduplikasi run yang konfigurasinya sama (pasangan + backend + perangkat; run terakhir
    menimpa yang lama), lalu beri label unik. Run dengan pair_name sama tetapi backend atau
    perangkat berbeda (mis. BlazeFace via MediaPipe vs TFLite) tetap ditampilkan terpisah.
    """
    entries = {}
    for b in benchmarks:
        backend = b.get("model_info", {}).get("backend", "")
        device = b.get("device_info", {}).get("hostname", "")
        entries[(b["pair_name"], backend, device)] = b["scenarios"]

    backends_per_pair = {}
    devices = set()
    for pair, backend, device in entries:
        backends_per_pair.setdefault(pair, set()).add(backend)
        devices.add(device)

    labeled = {}
    for (pair, backend, device), scenarios in entries.items():
        label = pair
        if len(backends_per_pair[pair]) > 1:
            # Ambil runtime deteksi saja, mis. "TFLite Interpreter (ai_edge_litert)"
            label += f" [{backend.split(' + ')[0]}]"
        if len(devices) > 1:
            label += f" @ {device}"
        labeled[label] = scenarios
    return labeled


def generate_comparison_charts(json_filepath, show=True):
    # 1. Memuat data dari file JSON
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {json_filepath} tidak ditemukan.")
        return

    print(f"Membaca: {json_filepath}")

    # 2. Ekstraksi model
    models_data = build_model_entries(data.get("benchmarks", []))
    if not models_data:
        print("Tidak ada data benchmark di file tersebut.")
        return
    models = list(models_data.keys())

    # Gabungan semua skenario, urutan mengikuti kemunculan pertama
    scenarios = []
    for sc in models_data.values():
        for s in sc:
            if s not in scenarios:
                scenarios.append(s)

    # Info perangkat untuk judul (jika semua run dari satu perangkat)
    device_types = {b.get("device_info", {}).get("device_type", "") for b in data.get("benchmarks", [])}
    device_str = device_types.pop() if len(device_types) == 1 else ""

    # 3. Mendefinisikan metrik yang akan divisualisasikan (nilai, apakah persen)
    metrics = {
        "Face Detection Rate (%)": (lambda sc: sc["detection_rate"], True),
        "Face Recognition Rate (%)": (lambda sc: sc["recognition_rate"], True),
        "Average Total Latency (ms)": (lambda sc: sc["total_pipeline_latency"]["mean"], False),
        "Effective FPS": (lambda sc: sc["effective_fps"], False),
    }

    def get_value(model, scenario, extractor):
        sc = models_data[model].get(scenario)
        return extractor(sc) if sc else np.nan

    # 4. Membuat subplot 2x2 untuk 4 grafik
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    x = np.arange(len(scenarios))
    group_width = 0.8                      # sisakan 0.2 jarak antar grup skenario
    width = group_width / len(models)      # lebar bar menyesuaikan jumlah model
    label_fontsize = 9 if len(models) <= 3 else 7
    label_rotation = 0 if len(models) <= 3 else 90

    # Warna khusus untuk setiap model agar konsisten di semua grafik
    colors = plt.get_cmap('tab10').colors

    for idx, (metric_title, (extractor, is_percent)) in enumerate(metrics.items()):
        ax = axes[idx]
        max_val = 0.0

        for i, model in enumerate(models):
            values = [get_value(model, s, extractor) for s in scenarios]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model, color=colors[i % len(colors)])

            # Label angka di atas setiap bar (dilewati jika data skenario tidak ada)
            for bar, v in zip(bars, values):
                if np.isnan(v):
                    continue
                max_val = max(max_val, v)
                ax.annotate(f'{v:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, v),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=label_fontsize, rotation=label_rotation)

        # Ruang kosong di atas bar agar label angka tidak menabrak judul/frame
        headroom = 1.12 if label_rotation == 0 else 1.25
        if is_percent:
            ax.set_ylim(0, max(100, max_val) * headroom)
        else:
            ax.set_ylim(0, max_val * headroom if max_val > 0 else 1)

        # Penyesuaian tampilan grafik
        ax.set_title(metric_title, fontsize=14, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

    # Satu legenda bersama di bawah gambar agar tidak menutupi bar
    handles, labels = axes[0].get_legend_handles_labels()
    ncol = len(models) if len(models) <= 3 else 2
    legend_rows = int(np.ceil(len(models) / ncol))
    fig.legend(handles, labels, loc='lower center', ncol=ncol, fontsize=11, frameon=False)

    title = "Benchmark Comparison: Model Performance Across Scenarios"
    if device_str:
        title += f" ({device_str})"
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.985)

    # Sisakan ruang bawah untuk legenda dan atas untuk judul
    bottom = 0.03 + 0.03 * legend_rows
    fig.tight_layout(rect=[0, bottom, 1, 0.95], h_pad=3.0, w_pad=2.0)

    # Simpan PNG di folder yang sama dengan JSON, dengan timestamp yang sama
    base = os.path.splitext(os.path.basename(json_filepath))[0].replace('comparison', 'benchmark_comparison_charts')
    output_filename = os.path.join(os.path.dirname(os.path.abspath(json_filepath)), f"{base}.png")
    fig.savefig(output_filename, dpi=300)
    print(f"Grafik perbandingan berhasil disimpan sebagai '{output_filename}'")

    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buat grafik dari comparison_*.json")
    parser.add_argument('json_file', nargs='?', default=None,
                        help="Path comparison_*.json (Default: file terbaru di results/)")
    parser.add_argument('--results-dir', default=RESULTS_DIR,
                        help=f"Folder pencarian comparison_*.json (Default: {RESULTS_DIR})")
    parser.add_argument('--no-show', action='store_true', help="Jangan tampilkan jendela grafik")
    args = parser.parse_args()

    json_file = args.json_file or find_latest_comparison(args.results_dir)
    if json_file is None:
        print(f"Tidak ada comparison_*.json di {args.results_dir}. Jalankan compare_results.py dulu.")
    else:
        generate_comparison_charts(json_file, show=not args.no_show)
