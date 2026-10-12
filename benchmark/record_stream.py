"""
Script untuk merekam MJPEG stream dari ESP32-CAM ke file video lokal (MP4/AVI).
Digunakan agar benchmark bisa mendapatkan sumber video yang persis sama.
"""

import cv2
import numpy as np
import time
import argparse
import urllib.request
import sys

def record_stream(url, output_file, duration, fps_target=10):
    print(f"Menginisialisasi perekaman dari: {url}")
    print(f"Menyimpan ke: {output_file}")
    print(f"Durasi: {duration} detik")
    
    try:
        stream = urllib.request.urlopen(url)
    except Exception as e:
        print(f"Gagal terhubung ke {url}: {e}")
        sys.exit(1)
        
    bytes_buffer = b''
    video_writer = None
    
    start_time = time.time()
    frames_recorded = 0
    
    print("Mulai merekam...")
    
    while time.time() - start_time < duration:
        try:
            bytes_buffer += stream.read(4096)
            a = bytes_buffer.find(b'\xff\xd8')
            b = bytes_buffer.find(b'\xff\xd9', a) if a != -1 else -1
            
            if a != -1 and b != -1 and b > a:
                last_b = bytes_buffer.rfind(b'\xff\xd9')
                last_a = bytes_buffer.rfind(b'\xff\xd8', 0, last_b)
                
                if last_a != -1 and last_b != -1 and last_b > last_a:
                    jpg_data = bytes_buffer[last_a:last_b + 2]
                    bytes_buffer = bytes_buffer[last_b + 2:]
                else:
                    jpg_data = bytes_buffer[a:b + 2]
                    bytes_buffer = bytes_buffer[b + 2:]
                    
                frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    if video_writer is None:
                        h, w = frame.shape[:2]
                        # Gunakan mp4v codec untuk mp4
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(output_file, fourcc, fps_target, (w, h))
                        print(f"Resolusi video: {w}x{h}")
                        
                        # Buat window resizable agar ukurannya bisa disesuaikan
                        cv2.namedWindow("ESP32-CAM Recording", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("ESP32-CAM Recording", w, h)
                        print("Tekan 'q' pada jendela GUI untuk berhenti merekam lebih awal.")
                        
                    video_writer.write(frame)
                    frames_recorded += 1
                    
                    # Tambahkan teks informasi durasi ke frame yang ditampilkan di layar
                    elapsed = time.time() - start_time
                    disp_frame = frame.copy()
                    cv2.putText(disp_frame, f"REC: {elapsed:.1f}s / {duration}s", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imshow("ESP32-CAM Recording", disp_frame)
                    
                    # Tunggu input keyboard (1ms). Jika 'q' ditekan, hentikan rekam.
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nDihentikan manual oleh user.")
                        break
                    
                    sys.stdout.write(f"\rMerekam: {elapsed:.1f}s / {duration}s | Frame: {frames_recorded}")
                    sys.stdout.flush()
                    
        except Exception as e:
            print(f"\nError saat membaca stream: {e}")
            break
            
    print("\nSelesai merekam.")
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
        
    print(f"Video tersimpan: {output_file} ({frames_recorded} frame)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rekam stream ESP32-CAM ke video lokal")
    parser.add_argument("--url", default="http://10.7.28.33:81/stream", help="URL stream (default: http://10.7.28.33:81/stream)")
    parser.add_argument("--output", default="recording.mp4", help="File output (default: recording.mp4)")
    parser.add_argument("--duration", type=int, default=60, help="Durasi rekaman dalam detik (default: 60)")
    parser.add_argument("--fps", type=int, default=15, help="Target FPS (default: 15)")
    args = parser.parse_args()
    
    record_stream(args.url, args.output, args.duration, args.fps)
