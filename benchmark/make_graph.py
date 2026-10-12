import json
import numpy as np
import matplotlib.pyplot as plt

def generate_comparison_charts(json_filepath):
    # 1. Memuat data dari file JSON
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File {json_filepath} tidak ditemukan.")
        return

    # 2. Ekstraksi dan deduplikasi model
    # Menggunakan dictionary untuk menimpa duplikat (seperti YuNet yang muncul dua kali di JSON)
    models_data = {}
    for benchmark in data.get("benchmarks", []):
        pair_name = benchmark["pair_name"]
        models_data[pair_name] = benchmark["scenarios"]

    models = list(models_data.keys())
    # Mengambil daftar skenario dari model pertama (frontal, tilt_left, tilt_right, far_distance, low_light)
    scenarios = list(models_data[models[0]].keys())

    # 3. Mendefinisikan metrik yang akan divisualisasikan
    metrics = {
        "Face Detection Rate (%)": lambda m, s: models_data[m][s]["detection_rate"],
        "Face Recognition Rate (%)": lambda m, s: models_data[m][s]["recognition_rate"],
        "Average Total Latency (ms)": lambda m, s: models_data[m][s]["total_pipeline_latency"]["mean"],
        "Effective FPS": lambda m, s: models_data[m][s]["effective_fps"]
    }

    # 4. Membuat subplot 2x2 untuk 4 grafik
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    x = np.arange(len(scenarios))
    width = 0.25  # Lebar bar disesuaikan dengan jumlah model (3 model)

    # Warna khusus untuk setiap model agar konsisten di semua grafik
    colors = ['#4C72B0', '#DD8452', '#55A868']

    for idx, (metric_title, extractor) in enumerate(metrics.items()):
        ax = axes[idx]
        
        for i, model in enumerate(models):
            # Mengambil nilai untuk setiap skenario pada model tertentu
            values = [extractor(model, s) for s in scenarios]
            
            # Menghitung offset agar posisi bar sejajar berdampingan
            offset = (i - len(models)/2 + 0.5) * width
            
            # Menggambar bar chart
            bars = ax.bar(x + offset, values, width, label=model, color=colors[i % len(colors)])
            
            # Menambahkan label angka di atas setiap bar
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # Offset vertikal 3 poin
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        # Penyesuaian tampilan grafik
        ax.set_title(metric_title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Menambahkan legenda hanya pada grafik pertama agar tidak memenuhi layar
        if idx == 0:
            ax.legend(fontsize=10, loc='lower left')

    # Merapikan jarak antar grafik dan menyimpan hasilnya
    plt.suptitle("Benchmark Comparison: Model Performance Across Scenarios", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_filename = "benchmark_comparison_charts.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Grafik perbandingan berhasil disimpan sebagai '{output_filename}'")
    
    # Menampilkan grafik di layar
    plt.show()

if __name__ == "__main__":
    generate_comparison_charts("results/comparison_20261007_105637.json")