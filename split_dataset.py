import os
import shutil
import csv

# ============================================================
# Script Gabungan Dataset — FIXED VERSION
# 
# Bug sebelumnya:
#   1. Bagian Raffi mencari folder cctv/kanan/kiri yang tidak ada
#      di HPC (dataset_raffi sudah berbentuk train/val/test)
#   2. Path di CSV relative, bukan absolute — sweep code butuh path penuh
#
# Fix:
#   1. Baca Raffi dari struktur train/val/test yang sudah ada
#   2. Tulis absolute path di CSV
# ============================================================

# --- PATH CONFIG ---
root_path_raffi = "/home/coder/data_skripsi/dataset_raffi"
root_path_mutia = "/home/coder/data_skripsi/dataset_teh_mutia"
output_path     = "/home/coder/data_skripsi/dataset_gabungan_siap_training"

kelas_aksi = [
    '1_mengangguk', '2_mengangkat_tangan',
    '3_menggunakan_hp', '4_menopang_kepala', '5_menunduk'
]
splits = ['train', 'val', 'test']

# --- 1. BUAT STRUKTUR FOLDER OUTPUT ---
for split in splits:
    for kelas in kelas_aksi:
        os.makedirs(os.path.join(output_path, split, kelas), exist_ok=True)

# Tempat menampung baris CSV: [absolute_video_path, label]
csv_records = {'train': [], 'val': [], 'test': []}

# ============================================================
# LANGKAH 1: Salin data TEH MUTIA
# Struktur: dataset_teh_mutia/train|val|test/kelas/video.mp4
# ============================================================
print("=== LANGKAH 1: Menyalin data Teh Mutia ===")
for split in splits:
    count = 0
    for kelas in kelas_aksi:
        folder_mutia = os.path.join(root_path_mutia, split, kelas)
        if not os.path.exists(folder_mutia):
            print(f"  ⚠ Folder tidak ditemukan: {folder_mutia}")
            continue

        for fname in os.listdir(folder_mutia):
            if not fname.lower().endswith(('.mp4', '.avi')):
                continue

            path_asli  = os.path.join(folder_mutia, fname)
            nama_baru  = f"mutia_{fname}"
            path_tujuan = os.path.join(output_path, split, kelas, nama_baru)

            shutil.copy2(path_asli, path_tujuan)

            # Absolute path — yang sweep code butuhkan
            csv_records[split].append([path_tujuan, kelas])
            count += 1

    print(f"  ✅ {split}: {count} video Teh Mutia disalin")

# ============================================================
# LANGKAH 2: Salin data RAFFI
# Struktur: dataset_raffi/train|val|test/kelas/video.mp4
# (sudah di-split, tidak perlu random split lagi)
# ============================================================
print("\n=== LANGKAH 2: Menyalin data Raffi ===")
for split in splits:
    count = 0
    for kelas in kelas_aksi:
        folder_raffi = os.path.join(root_path_raffi, split, kelas)
        if not os.path.exists(folder_raffi):
            print(f"  ⚠ Folder tidak ditemukan: {folder_raffi}")
            continue

        for fname in os.listdir(folder_raffi):
            if not fname.lower().endswith(('.mp4', '.avi')):
                continue

            path_asli   = os.path.join(folder_raffi, fname)
            nama_baru   = f"raffi_{fname}"
            path_tujuan = os.path.join(output_path, split, kelas, nama_baru)

            shutil.copy2(path_asli, path_tujuan)

            # Absolute path
            csv_records[split].append([path_tujuan, kelas])
            count += 1

    print(f"  ✅ {split}: {count} video Raffi disalin")

# ============================================================
# LANGKAH 3: Buat CSV metadata
# Kolom: video_path, label
# (sesuai dengan yang sweep code ekspektasikan)
# ============================================================
print("\n=== LANGKAH 3: Membuat CSV metadata ===")
for split in splits:
    csv_path = os.path.join(output_path, f"{split}_metadata.csv")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video_path', 'label'])
        writer.writerows(csv_records[split])
    print(f"  📄 {split}_metadata.csv — {len(csv_records[split])} video")

# ============================================================
# LANGKAH 4: Verifikasi distribusi
# ============================================================
print("\n=== VERIFIKASI DISTRIBUSI ===")
for split in splits:
    records = csv_records[split]
    total = len(records)
    print(f"\n{split.upper()} — Total: {total}")
    for kelas in kelas_aksi:
        jumlah = sum(1 for r in records if r[1] == kelas)
        print(f"  {kelas}: {jumlah}")

print(f"\n✅ Selesai! Dataset gabungan tersimpan di: {output_path}")