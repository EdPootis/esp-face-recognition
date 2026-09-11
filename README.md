# ESP32-CAM / ESP32-S3 Face Recognition & 2-Factor Attendance System

Sistem pengenalan wajah (Face Recognition) dan verifikasi absensi dua faktor (2-Factor Authentication) berbasis Computer Vision menggunakan model OpenCV YuNet dan SFace. Sistem ini terintegrasi langsung dengan mikrokontroler ESP32-CAM / ESP32-S3, modul Card Reader NFC/RFID (PN532), sensor gerak PIR (HC-SR501), layar LCD 16x2 I2C, serta backend absensi Django.

---

## 1. Daftar dan Fungsi Setiap File Script

### a. `ktm_face_verification.py`
Skrip utama untuk sistem verifikasi absensi 2-Faktor (2FA):
- **Alur Kerja**: Mahasiswa menempelkan Kartu Tanda Mahasiswa (KTM) pada Card Reader -> Sistem menerima UID kartu -> Sistem mencocokkan wajah live di depan kamera dengan profil foto pemilik kartu (1-to-1 Verification).
- **Fitur Utama**:
  - Webhook Server HTTP internal (port 5050) untuk menerima notifikasi tap kartu dari ESP32 secara instan.
  - Jendela waktu verifikasi (default 8 detik) dengan visual countdown timer.
  - Deteksi kecurangan / joki absensi: Jika kartu milik mahasiswa A ditempelkan namun yang berdiri di depan kamera adalah mahasiswa B, sistem akan menolak akses dan memberi peringatan.
  - Pengiriman log kehadiran terverifikasi ke Django Dashboard API dan notifikasi ke LCD ESP32.
  - Fitur simulasi tap kartu manual dengan menekan tombol `t` pada keyboard untuk pengujian tanpa hardware reader.

### b. `multi_esp32cam.py`
Skrip pengenalan wajah multi-kamera secara simultan (1-to-N Recognition):
- Menghubungkan beberapa kamera ESP32 sekaligus (misalnya ESP32-S3, ESP32-CAM 1, ESP32-CAM 2) dalam satu tampilan antarmuka grid (tata letak berdampingan / multi-tile).
- Menggunakan arsitektur multithreading dan HTTP keep-alive buffer untuk meminimalkan latensi video (0-lag).
- Menampilkan bounding box deteksi wajah secara real-time dan nama hasil identifikasi.
- Mengirimkan log deteksi ke Django Dashboard dan menampilkan status pengenalan pada layar LCD ESP32.

### c. `esp32cam.py`
Skrip pengenalan wajah untuk satu kamera (Single Camera):
- Menghubungkan satu kamera ESP32 via HTTP `/capture` atau `/stream`.
- Melakukan enrollment wajah dari folder `known_faces/`, deteksi wajah dengan YuNet, dan pengenalan fitur dengan SFace.
- Cocok digunakan untuk pengujian cepat unit kamera tunggal.

### d. `students_database.json`
File konfigurasi database pemetaan kartu mahasiswa:
- Menyimpan relasi antara UID fisik kartu NFC/RFID, NIM, Nama Mahasiswa, Jurusan, dan nama folder foto referensi di `known_faces/`.
- Memungkinkan penggantian kartu yang hilang/rusak tanpa perlu mengubah struktur folder atau melakukan re-index foto wajah.

### e. File Model AI (`.onnx`)
- `face_detection_yunet_2023mar.onnx`: Model deep learning ringan untuk deteksi posisi wajah dan landmark (mata, hidung, mulut) secara cepat.
- `face_recognition_sface_2021dec.onnx`: Model ekstraksi fitur wajah (128-dimensional embedding) untuk membandingkan kemiripan wajah menggunakan Cosine Similarity.

---

## 2. Struktur Direktori Proyek

```text
esp32cam face recognition/
|-- known_faces/                    # Direktori foto wajah referensi per orang
|   |-- edmond/
|   |   |-- foto1.jpg
|   |   `-- foto2.jpg
|   |-- nama_orang_2/
|   |   `-- foto1.jpg
|   `-- nama_orang_3/
|       `-- foto1.jpg
|-- face_detection_yunet_2023mar.onnx   # Model deteksi wajah YuNet
|-- face_recognition_sface_2021dec.onnx # Model pengenalan wajah SFace
|-- students_database.json             # Database pemetaan UID KTM ke profil
|-- ktm_face_verification.py           # Skrip absensi 2FA (KTM + Face Recognition)
|-- multi_esp32cam.py                  # Skrip monitoring multi-kamera grid
|-- esp32cam.py                        # Skrip pengenalan single-kamera
|-- requirements.txt                   # Daftar dependensi Python
`-- README.md                          # Dokumentasi proyek
```

---

## 3. Prasyarat dan Instalasi

### Persyaratan Lingkungan
- Python versi 3.8 atau lebih baru.
- Koneksi jaringan Wi-Fi lokal yang sama antara komputer dan ESP32.

### Langkah Instalasi Dependensi
Jalankan perintah berikut di terminal:

```bash
pip install -r requirements.txt
```

Atau instalasi paket secara manual:
```bash
pip install opencv-python numpy pillow urllib3
```

---

## 4. Konfigurasi Sistem

### a. Menambahkan Wajah Baru
1. Buat folder baru di dalam direktori `known_faces/` dengan nama orang terkait (contoh: `known_faces/edmond/`).
2. Masukkan satu atau beberapa foto wajah yang jelas (format `.jpg`, `.jpeg`, atau `.png`) ke dalam folder tersebut.

### b. Menghubungkan Kartu KTM ke Profil Mahasiswa
Buka file `students_database.json` dan tambahkan data UID kartu:

```json
{
  "students": {
    "DEA35B89": {
      "nim": "13520001",
      "name": "Edmond",
      "folder": "edmond",
      "major": "Informatika"
    }
  }
}
```
*Catatan: Nilai `folder` harus sesuai dengan nama folder di dalam `known_faces/`.*

### c. Mengatur IP Kamera dan Server
- Di dalam file `ktm_face_verification.py` dan `multi_esp32cam.py`, sesuaikan dictionary `CAMERAS` dengan alamat IP ESP32 Anda:
  ```python
  CAMERAS = {
      'ESP32-S3': 'http://10.66.53.60/capture',
      'ESP32-CAM 1': 'http://10.244.226.130/capture',
  }
  ```
- Sesuaikan `DASHBOARD_URL` dengan endpoint server Django jika menggunakan pencatatan log otomatis.

---

## 5. Panduan Menjalankan Program

### a. Menjalankan Mode Verifikasi Absensi 2-Faktor (KTM + Wajah)
Gunakan mode ini untuk alur absensi operasional:

```bash
python ktm_face_verification.py
```

**Cara Penggunaan:**
1. Sistem akan masuk ke status **STANDBY** (menunggu tap kartu).
2. Mahasiswa menempelkan kartu KTM pada modul reader NFC PN532.
3. Sistem berpindah ke status **VERIFYING** (durasi 8 detik). Mahasiswa melihat ke arah kamera.
4. Sistem memverifikasi kecocokan wajah dengan data kartu:
   - **Sukses**: Akses diterima, log absensi tercatat, layar LCD menampilkan konfirmasi nama.
   - **Wajah Tidak Cocok**: Akses ditolak dan peringatan kecurangan dicatat.
   - **Timeout**: Waktu habis jika wajah tidak terdeteksi.
5. Untuk simulasi pengujian tanpa reader fisik, tekan tombol **`t`** pada jendela GUI untuk memicu tap kartu contoh.
6. Tekan tombol **`q`** untuk keluar.

### b. Menjalankan Mode Multi-Kamera Grid
Gunakan mode ini untuk memonitor beberapa kamera secara bersamaan:

```bash
python multi_esp32cam.py
```

Argumen tambahan yang tersedia:
- `python multi_esp32cam.py --mode stream` : Menggunakan mode video streaming MJPEG.
- `python multi_esp32cam.py --mode capture` : Menggunakan mode snapshot polling (default, stabil).
- Tekan **`q`** pada jendela tampilan untuk menghentikan program.

### c. Menjalankan Mode Single Kamera
Gunakan mode ini untuk pengujian kamera tunggal:

```bash
python esp32cam.py
```

---

## 6. Protokol Komunikasi HTTP Webhook (ESP32 ke Python)

Saat kartu KTM di-tap pada hardware reader, ESP32 mengirimkan request HTTP GET ke webhook server lokal yang berjalan di `ktm_face_verification.py`:

```http
GET http://<IP_KOMPUTER>:5050/tap?uid=<KODE_UID>&cam=ESP32-S3
```

Parameter:
- `uid`: Kode unik kartu dalam format Hexadecimal (contoh: `DEA35B89`).
- `cam`: Identifier nama kamera yang bersangkutan.

Ketika verifikasi berhasil, skrip Python mengirimkan balik konfirmasi ke ESP32 melalui endpoint:
```http
GET http://<IP_ESP32>/face?name=<NAMA>&score=<SKOR>&cam=<KAMERA>&time=<WAKTU>
```
Sehingga layar LCD 16x2 pada ESP32 dapat langsung menampilkan informasi nama mahasiswa dan waktu absensi.
