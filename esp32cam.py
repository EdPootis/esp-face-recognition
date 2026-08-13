import os
import sys
import cv2
import urllib.request
import urllib.error
import numpy as np

if sys.platform == 'win32':
    import msvcrt

# Ganti dengan IP Address ESP32-CAM Anda yang sesuai
url = 'http://10.17.13.130/capture'

# Path dan URL model Face Detection YuNet (OpenCV 5 DNN)
model_path = 'face_detection_yunet_2023mar.onnx'
model_url = 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'

# Unduh model secara otomatis jika belum ada di direktori
if not os.path.exists(model_path):
    print("Mengunduh model face detection YuNet...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model berhasil diunduh.")

# Inisialisasi FaceDetectorYN (YuNet)
face_detector = cv2.FaceDetectorYN_create(
    model=model_path,
    config='',
    input_size=(320, 240),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

print("Mulai menarik gambar dari ESP32-CAM...")
print("Tekan 'q' pada jendela tampilan (GUI) ATAU di terminal untuk keluar.")

while True:
    try:
        # Cek apakah ada input 'q' dari terminal sebelum request
        if sys.platform == 'win32' and msvcrt.kbhit():
            if msvcrt.getch().lower() == b'q':
                print("\nTombol 'q' ditekan di terminal. Menutup program...")
                break

        # 1. Menarik gambar satu per satu dari endpoint /capture (dengan timeout 3 detik)
        with urllib.request.urlopen(url, timeout=3) as img_resp:
            # 2. Mengonversi data biner menjadi array NumPy
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)

        # 3. Decode data array menjadi frame gambar (format OpenCV)
        frame = cv2.imdecode(imgnp, -1)
        if frame is None:
            continue

        # 4. Sesuaikan input_size detector dengan ukuran frame yang diterima
        h, w, _ = frame.shape
        face_detector.setInputSize((w, h))

        # 5. Proses Face Detection menggunakan YuNet
        _, faces = face_detector.detect(frame)

        # 6. Gambar kotak di sekitar wajah yang terdeteksi
        if faces is not None:
            for face in faces:
                x, y, fw, fh = map(int, face[:4])
                
                # Pastikan koordinat berada di dalam batas frame
                x, y = max(0, x), max(0, y)
                fw, fh = min(fw, w - x), min(fh, h - y)

                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

                # Ekstraksi Region of Interest (ROI) wajah untuk tahap Recognition
                face_roi = frame[y:y+fh, x:x+fw]

                # --- LOGIKA FACE RECOGNITION MASUK DI SINI ---
                # Contoh pemanggilan: 
                # identity = my_recognition_model.predict(face_roi)
                # cv2.putText(frame, identity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 7. Tampilkan hasil pemrosesan di jendela komputer
        cv2.imshow("ESP32-CAM External Processing", frame)

        # Keluar jika tombol 'q' ditekan di jendela OpenCV atau di terminal
        key = cv2.waitKey(1) & 0xFF
        terminal_pressed_q = (sys.platform == 'win32' and msvcrt.kbhit() and msvcrt.getch().lower() == b'q')
        
        if key == ord('q') or terminal_pressed_q:
            print("\nKeluar dari program...")
            break

    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna (Ctrl+C).")
        break
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"Gagal mengambil gambar dari ESP32-CAM ({e}). Mencoba lagi...")
        # Periksa input 'q' dari terminal saat gagal koneksi
        if sys.platform == 'win32' and msvcrt.kbhit():
            if msvcrt.getch().lower() == b'q':
                print("\nTombol 'q' ditekan di terminal. Menutup program...")
                break
        continue
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        break

# Bersihkan resource saat selesai
cv2.destroyAllWindows()

