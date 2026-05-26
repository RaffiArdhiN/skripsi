import os
import cv2

# --- PENGATURAN PATH ---
# Kamu bisa ganti path ini ke dataset_raffi, dataset_teh_mutia, atau folder gabungan
root_path = "/home/coder/data_skripsi/dataset_teh_mutia"

print("==========================================================")
print("     PENGECEKAN DURASI VIDEO (< 3 DETIK ATAU > 9 DETIK)   ")
print("==========================================================\n")

# Variabel untuk rekapitulasi
total_bermasalah = 0
rekap_per_kelas = {}
detail_video_bermasalah = []

print("Sedang memindai dan menghitung durasi video... (Ini mungkin memakan waktu)\n")

# os.walk akan menelusuri semua folder dan subfolder secara otomatis
for root, dirs, files in os.walk(root_path):
    # Mencari nama kelas dari nama folder saat ini
    # Asumsinya folder kelas bernama seperti '1_mengangguk', dll
    nama_folder = os.path.basename(root)
    
    for file in files:
        if file.lower().endswith('.mp4'):
            file_path = os.path.join(root, file)
            
            # --- MENGHITUNG DURASI DENGAN OPENCV ---
            cap = cv2.VideoCapture(file_path)
            
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release() # Jangan lupa di-release agar memori tidak penuh
                
                # Menghindari error pembagian dengan nol
                if fps > 0:
                    durasi_detik = frame_count / fps
                else:
                    durasi_detik = 0
                
                # --- LOGIKA PENGECEKAN 3 - 9 DETIK ---
                if durasi_detik < 3.0 or durasi_detik > 9.0:
                    total_bermasalah += 1
                    
                    # Tambahkan ke rekap per kelas
                    if nama_folder not in rekap_per_kelas:
                        rekap_per_kelas[nama_folder] = 0
                    rekap_per_kelas[nama_folder] += 1
                    
                    # Simpan detailnya untuk ditampilkan nanti
                    status = "Kekurangan Durasi" if durasi_detik < 3.0 else "Kelebihan Durasi"
                    detail_video_bermasalah.append(f"[{status}] {nama_folder}/{file} -> {durasi_detik:.2f} detik")
            else:
                print(f"⚠️ Gagal membaca video: {file_path}")

# --- MENAMPILKAN HASIL ---
if total_bermasalah > 0:
    print("DAFTAR KELAS DENGAN VIDEO TIDAK MEMENUHI STANDAR (3-9 Detik):")
    for kelas, jumlah in sorted(rekap_per_kelas.items()):
        print(f"   └── 📁 {kelas}: {jumlah} video bermasalah")
        
    print("\n----------------------------------------------------------")
    print("DETAIL VIDEO:")
    # Menampilkan maksimal 20 detail pertama agar terminal tidak terlalu penuh
    for detail in detail_video_bermasalah[:20]:
        print(f" - {detail}")
        
    if len(detail_video_bermasalah) > 20:
        print(f"   ... dan {len(detail_video_bermasalah) - 20} video lainnya.")
else:
    print("🎉 Keren! Semua video sudah berada di rentang 3 - 9 detik.")

print("\n==========================================================")
print(f" TOTAL KESELURUHAN VIDEO BERMASALAH: {total_bermasalah} video")
print("==========================================================")