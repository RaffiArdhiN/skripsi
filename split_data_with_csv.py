import os
import random
import shutil
import csv

# --- PENGATURAN PATH ---
root_path = r"E:\data skripsi\split_dataset_raffi"
output_path = r"E:\data skripsi\dataset_raffi"

# Rasio Pembagian (Train 80%, Val 10%, Test 10%)
train_ratio = 0.8
val_ratio = 0.1
# Sisanya otomatis jadi test_ratio

# Daftar sumber angle kamera dan nama kelas
sumber_kamera = ['cctv', 'kanan', 'kiri']
kelas_aksi = [
    '1_mengangguk', '2_mengangkat_tangan', 
    '3_menggunakan_hp', '4_menopang_kepala', '5_menunduk'
]

# --- 1. MEMBUAT STRUKTUR FOLDER OUTPUT ---
for split in ['train', 'val', 'test']:
    for kelas in kelas_aksi:
        os.makedirs(os.path.join(output_path, split, kelas), exist_ok=True)

print("Mulai memproses, membagi dataset, dan menyusun metadata...\n")

# Dictionary untuk menampung data CSV per split
metadata_records = {'train': [], 'val': [], 'test': []}

# --- 2. MENGUMPULKAN DAN MEMBAGI DATA PER KELAS ---
for kelas in kelas_aksi:
    semua_video_kelas = []
    
    # Ambil index kelas sebagai label_id (0, 1, 2, 3, 4) untuk kebutuhan training
    label_id = kelas_aksi.index(kelas)
    
    # Kumpulkan video dari cctv, kanan, dan kiri untuk kelas ini
    for sumber in sumber_kamera:
        folder_sumber = os.path.join(root_path, sumber, kelas)
        
        if os.path.exists(folder_sumber):
            # Ambil file .mp4
            videos = [f for f in os.listdir(folder_sumber) if f.lower().endswith('.mp4')]
            for vid in videos:
                path_asli = os.path.join(folder_sumber, vid)
                # Format nama baru agar tidak ada yang tertimpa
                nama_baru = f"{sumber}_{vid}"
                semua_video_kelas.append((path_asli, nama_baru))
                
    # PENTING: Urutkan abjad terlebih dahulu agar hasil acakan random.seed SELALU konsisten
    semua_video_kelas.sort(key=lambda x: x[1])
    
    # Acak urutan video agar distribusinya merata
    random.seed(42) 
    random.shuffle(semua_video_kelas)
    
    # Hitung jumlah untuk tiap split
    total_vid = len(semua_video_kelas)
    train_count = int(total_vid * train_ratio)
    val_count = int(total_vid * val_ratio)
    
    # Potong list berdasarkan hitungan
    train_data = semua_video_kelas[:train_count]
    val_data = semua_video_kelas[train_count:train_count + val_count]
    test_data = semua_video_kelas[train_count + val_count:]
    
    # --- 3. COPY FILE KE FOLDER MASING-MASING & CATAT METADATA ---
    distribusi = {'train': train_data, 'val': val_data, 'test': test_data}
    
    for split_name, data in distribusi.items():
        for path_asli, nama_baru in data:
            path_tujuan = os.path.join(output_path, split_name, kelas, nama_baru)
            shutil.copy2(path_asli, path_tujuan)
            
            # Format path relatif di dalam folder split (contoh: "1_mengangguk/cctv_video1.mp4")
            # Menggunakan forward-slash (/) agar aman dibaca script deep learning di Linux/HPC
            relative_path = f"{kelas}/{nama_baru}"
            
            # Masukkan data ke rekaman metadata: [path_video, nama_kelas, id_numerik]
            metadata_records[split_name].append([relative_path, kelas, label_id])
            
    print(f"✅ Kelas {kelas}: Total {total_vid} video -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# --- 4. MEMBUAT FILE CSV METADATA ---
print("\n==========================================================")
print("              PEMBUATAN FILE CSV METADATA                 ")
print("==========================================================")

for split_name, rows in metadata_records.items():
    csv_file_path = os.path.join(output_path, f"{split_name}_metadata.csv")
    
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Menulis nama kolom header
        writer.writerow(['video_path', 'label_name', 'label_id'])
        # Menulis seluruh baris data video
        writer.writerows(rows)
        
    print(f"📄 Berhasil membuat: {split_name}_metadata.csv ({len(rows)} baris)")

print(f"\nSelesai! Seluruh file video dan 3 file CSV metadata disimpan di: {output_path}")