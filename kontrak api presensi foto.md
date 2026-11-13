# Kontrak API: Log Presensi Mode KTM + Foto (`POST /api/log/`)

| | |
| :--- | :--- |
| **Penyedia** | Dashboard Presentronik Fasilkom (`POST /api/log/`) |
| **Pemakai** | `ktm_presensi_foto.py` (repo `esp32cam-face-recognition`), dipasangkan dengan firmware `ktm_foto_s3` |
| **Versi** | 1 — 13 November 2026 |
| **Kontak** | Edmond Christian (skrip & firmware), Juan Lukius Barnaby (dashboard) |

## 1. Tujuan

Mode eksperimen usulan pembimbing. **Tap KTM saja sudah dihitung sebagai kehadiran**, seperti Presentronik lama. Kamera hanya mengambil satu foto saat tap untuk dokumentasi. Tidak ada face recognition, dan presensi tetap dikirim walau foto gagal diambil.

Satu node menjalankan salah satu mode saja: face recognition (`ktm_fr_s3` + `ktm_face_verification*.py`) atau KTM + Foto (`ktm_foto_s3` + `ktm_presensi_foto.py`).

## 2. Endpoint

Sama dengan mode face recognition. Tidak ada endpoint atau field baru.

| | |
| :--- | :--- |
| **Log** | `POST http://<host-dashboard>:8000/api/log/`, body JSON, header `X-API-Key` |
| **Daftar kamera** | `GET /api/kamera/` saat start, sesuai `kontrak api camera info dashboard.md` |

## 3. Nilai Field di Mode Ini

Satu log per tap kartu. Firmware mengabaikan kartu yang sama selama 3 detik, jadi kartu yang ditahan di reader tidak menghasilkan log ganda.

| Field | Nilai di mode KTM + Foto |
| :--- | :--- |
| `tipe_perangkat` | `ESP32-S3 KTM + Foto`. Penanda mode; dashboard mencocokkannya tanpa memperhatikan huruf besar/kecil dan spasi. |
| `status` | `HADIR` untuk kartu yang ada di database lokal skrip, `Unknown` untuk kartu lain. Tidak pernah `DITOLAK`, `TIMEOUT`, atau `MISMATCH`. |
| `no_identitas`, `nim` | NPM, atau `-` untuk kartu `Unknown` |
| `nama`, `person_name` | Nama mahasiswa, atau `Unknown` |
| `rfid_tag` | UID kartu, hex huruf besar tanpa spasi (mis. `DEA35B89`) |
| `kamera` | `NODE_NAME` node tempat kartu di-tap |
| `mac_address` | MAC asli board dari firmware. Tidak dikirim jika belum diketahui. |
| `ts_presensi_device` | Waktu tap (`YYYY-MM-DD HH:MM:SS`, waktu lokal komputer skrip). Tetap waktu tap asli walau log dikirim ulang (bagian 4). |
| `timestamp` | Waktu tap dalam ISO 8601 |
| `skor`, `similarity` | Selalu `0.0` (tidak ada pencocokan wajah) |
| `recognized` | `true` jika kartu ada di database lokal skrip. Diabaikan dashboard. |
| `foto_b64`, `image_base64` | Foto saat tap, `data:image/jpeg;base64,...` (640×480), atau `null` jika foto gagal diambil |
| `kd_org` | Sama dengan mode face recognition |

Perilaku dashboard yang sudah dikonfirmasi (13 November 2026):
- `HADIR` dengan NPM yang dikenal ditulis ke `log_presentronik` (format sama dengan tap KTM) dan diproses sinkron SIAK.
- Kartu `Unknown` tetap dicatat sebagai kehadiran jika UID-nya ada di data kartu SIAK. Karena itu LCD tidak menampilkan "kartu tidak dikenal", melainkan `KTM Tercatat`.
- Dashboard hanya memakai `status`. `recognized` diabaikan, dan tidak ada ambang skor.
- `foto_b64: null` diterima; kehadiran tetap tersimpan dan halaman menampilkan "Foto tidak ada".
- Foto disimpan sebagai file JPEG di server dashboard; database hanya menyimpan path-nya.
- Log tampil di halaman **Log Face Recognition** (filter mode "KTM + Foto") dan di tabel tersendiri pada **Laporan Kehadiran**.

## 4. Pengiriman Ulang

Jika `POST /api/log/` gagal karena dashboard tidak bisa dihubungi, timeout, HTTP 5xx, atau 401, skrip menyimpan payload utuh di folder `antrian_log/`. Payload lalu dikirim ulang di latar saat skrip start dan setiap 30 detik, terlama dulu, dengan `ts_presensi_device` asli. Antrian disimpan di disk, jadi tetap ada setelah skrip di-restart.

- **Duplikat mungkin terjadi**, misalnya jika dashboard sempat menyimpan log tetapi balasannya tidak sampai. Dashboard tetap menghitung kehadiran sekali per kelas; hanya daftar log yang mendapat baris tambahan.
- **HTTP 400, 413, dan 422** dianggap payload ditolak. Log seperti ini tidak diantrekan, dan log antrian yang ditolak dipindah ke `antrian_log/ditolak/` untuk diperiksa manual.

## 5. Contoh

```json
{
  "no_identitas": "2306208363", "nama": "Budi Santoso", "rfid_tag": "DEA35B89",
  "kd_org": "01.00.12.01", "tipe_perangkat": "ESP32-S3 KTM + Foto",
  "ts_presensi_device": "2026-11-13 08:01:12", "kamera": "ESP32-S3-1",
  "mac_address": "24:6F:28:AB:CD:EF", "skor": 0.0, "status": "HADIR",
  "recognized": true, "foto_b64": "data:image/jpeg;base64,/9j/...",
  "nim": "2306208363", "person_name": "Budi Santoso", "similarity": 0.0,
  "timestamp": "2026-11-13T08:01:12.345678", "image_base64": "data:image/jpeg;base64,/9j/..."
}
```

## 6. Pertanyaan Terbuka

- **Akses foto:** saat ini mengikuti aturan foto face recognition (admin semua, dosen kelasnya, mahasiswa miliknya sendiri). Menunggu keputusan pembimbing apakah dibatasi ke admin saja.
