# ============================================================
# UKUR FPS CPU-ONLY — Proof of Concept Edge Device
# ============================================================
# Latar belakang: proposal (BAB III) menyebut FPS diuji di workstation
# (GPU), tapi pengajuan HPC ke Pa Yaya menyebut komitmen tambahan:
# "pengukuran FPS di CPU sebagai proof of concept di edge device".
# Seluruh narasi proposal (Latar Belakang, Abstrak) berputar di sekitar
# klaim "model ringan untuk edge device" — klaim itu butuh data CPU,
# bukan cuma GPU kelas atas (RTX 5090), untuk benar-benar dibuktikan.
#
# Script ini TIDAK melatih apa-apa. Cuma load checkpoint yang SUDAH ADA
# (dari training GPU sebelumnya), forward pass di CPU saja (GPU
# di-nonaktifkan paksa), ukur throughput.
#
# DUA PILIHAN TEMPAT JALANKAN:
#   A) Di HPC, tapi GPU dipaksa nonaktif (DEVICE="cpu" eksplisit).
#      → Paling cepat & praktis: checkpoint, dataset, dan cache model
#        HuggingFace sudah ada di sana. Tetap valid secara metodologis
#        sebagai "CPU-only baseline" — umum dipakai di paper sejenis.
#   B) Di laptop ASUS Vivobook (i3-1005G1) yang SUDAH terdokumentasi
#      resmi di proposal sebagai "Perangkat Komputasi Lokal".
#      → Lebih otentik sebagai representasi edge device sungguhan,
#        tapi perlu pastikan dataset test + checkpoint + library
#        (torch, timm, transformers, opencv) sudah ada di laptop itu,
#        dan VideoMAE base weights bisa butuh download ulang dari
#        HuggingFace Hub kalau belum pernah di-cache di laptop tsb.
#
# Untuk deadline hari ini, opsi A direkomendasikan dulu (cepat, no
# friction). Opsi B bisa jadi data tambahan kalau masih ada waktu
# setelah opsi A selesai — sama-sama sah dipakai di BAB IV, beri
# label device-nya masing-masing di tabel agar jelas.
# ============================================================

import os, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
import cv2
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEForVideoClassification

# -------------------------
# Config
# -------------------------
DEVICE = "cpu"   # SENGAJA dipaksa, walau GPU tersedia di mesin ini

NUM_FRAMES     = 16
NUM_CLASSES    = 5
TEACHER_SIZE   = 224
STUDENT_SIZE   = 256

# Lebih kecil dari versi GPU — CPU jauh lebih lambat per-batch,
# batch kecil + worker=0 lebih stabil & cepat selesai di hardware lemah.
FPS_BATCH_SIZE = 2
FPS_WORKERS    = 0    # hindari overhead multiprocessing di CPU terbatas
N_WARM         = 2
N_RUN          = 10   # cukup untuk estimasi stabil, tidak butuh 20 seperti GPU
COOLDOWN_BETWEEN_MODELS = 2

# --- Pilih cakupan pengukuran ---
# "required_only" → 3 skenario WAJIB sesuai proposal (Teacher, Student
#                    no-KD, KD dengan T=11 & alpha=0.45 & mean — sesuai
#                    Tabel 3.4 proposal). Paling cepat, defensible secara
#                    minimum.
# "all"           → seluruh 26 skenario (Teacher + Student baseline +
#                    24 kombinasi KD). Lebih kaya untuk BAB IV, makan
#                    waktu lebih lama (CPU lambat, tapi MobileViT ringan
#                    jadi tetap masuk akal — estimasi totalnya akan
#                    diprint di awal sebelum mulai).
SCENARIOS_FILTER = "all"

TEST_CSV = "/home/coder/data_skripsi/dataset_gabungan_siap_training/test_metadata.csv"
CHECKPOINT_DIR = "/home/coder/output_model/skenario_raffi/checkpoints"
RESULTS_DIR    = "/home/coder/output_model/skenario_raffi/results"
RESULTS_CSV    = os.path.join(RESULTS_DIR, "experiment_results.csv")
SUMMARY_CPU_TXT = os.path.join(RESULTS_DIR, "summary_cpu.txt")

# Config KD yang WAJIB sesuai proposal Tabel 3.4 (T=11, alpha=0.45, mean)
REQUIRED_KD_CONFIG = "StandardKD_a=0.45_T=11.0"

# -------------------------
# Label & transform
# -------------------------
label_to_idx = {
    "1_mengangguk": 0, "2_mengangkat_tangan": 1, "3_menggunakan_hp": 2,
    "4_menopang_kepala": 3, "5_menunduk": 4
}
mean_norm = [0.485, 0.456, 0.406]
std_norm  = [0.229, 0.224, 0.225]

def make_transform(size):
    return T.Compose([
        T.ToPILImage(), T.Resize((size, size)),
        T.ToTensor(), T.Normalize(mean=mean_norm, std=std_norm)
    ])

class FPSDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.tf_teacher = make_transform(TEACHER_SIZE)
        self.tf_student = make_transform(STUDENT_SIZE)

    def __len__(self):
        return len(self.df)

    def _read_video(self, path):
        cap, frames = cv2.VideoCapture(path), []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        label  = label_to_idx[row['label']]
        frames = self._read_video(row['video_path'])
        if not frames:
            frames = [np.zeros((max(TEACHER_SIZE, STUDENT_SIZE),)*2+(3,),
                                dtype=np.uint8)] * NUM_FRAMES

        total   = len(frames)
        indices = np.linspace(0, total-1, NUM_FRAMES).astype(int) \
                  if total >= NUM_FRAMES \
                  else np.pad(np.linspace(0, total-1, total).astype(int),
                              (0, NUM_FRAMES - total), mode='wrap')
        sampled = [frames[i] for i in indices]

        t_teach = torch.stack([self.tf_teacher(f) for f in sampled])
        t_stud  = torch.stack([self.tf_student(f)  for f in sampled])
        return t_teach, t_stud, label

# -------------------------
# Model definitions — identik dengan train_classifier.py
# -------------------------
def build_teacher():
    m = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    return m.to(DEVICE)

class MobileViTVideo(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, temporal_type='mean'):
        super().__init__()
        self.temporal_type = temporal_type
        self.backbone      = timm.create_model('mobilevit_s', pretrained=True, num_classes=0)
        self.embed_dim     = self.backbone.num_features \
                             if hasattr(self.backbone, 'num_features') else 640

        if self.temporal_type == 'lstm':
            self.temporal_layer = nn.LSTM(
                input_size=self.embed_dim, hidden_size=256,
                num_layers=1, batch_first=True)
            self.classifier = nn.Linear(256, num_classes)
        else:
            self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        B, T_, C, H, W = x.shape
        feats = self.backbone(x.view(B*T_, C, H, W)).view(B, T_, -1)
        if self.temporal_type == 'lstm':
            _, (h_n, _) = self.temporal_layer(feats)
            pooled = h_n[-1]
        else:
            pooled = feats.mean(1)
        return self.classifier(pooled)

# -------------------------
# FPS measurement
# -------------------------
def measure_fps(model, loader, mode='student', n_warm=N_WARM, n_run=N_RUN):
    model.eval()
    it = iter(loader)

    def _fwd(tb, ts):
        return model(pixel_values=tb.to(DEVICE)) if mode == 'teacher' \
               else model(ts.to(DEVICE))

    def _next():
        nonlocal it
        try: return next(it)
        except StopIteration:
            it = iter(loader); return next(it)

    with torch.no_grad():
        for _ in range(n_warm):
            _fwd(*_next()[:2])

    t0, total_frames = time.time(), 0
    with torch.no_grad():
        for _ in range(n_run):
            tb, ts, _ = _next()
            _fwd(tb, ts)
            total_frames += tb.shape[0] * tb.shape[1]

    return total_frames / (time.time() - t0)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Tidak ditemukan: {RESULTS_CSV}")

    print(f"DEVICE dipaksa: {DEVICE} (GPU diabaikan walau tersedia)\n")
    print("Membuat test loader CPU (batch kecil, single-process)...")
    test_loader = DataLoader(
        FPSDataset(TEST_CSV),
        batch_size=FPS_BATCH_SIZE,
        shuffle=True,   # representatif lintas 5 kelas — CSV test berurutan per kelas,
                         # tanpa shuffle, n_run batch pertama bisa cuma kena 1-2 kelas saja
        num_workers=FPS_WORKERS,
        pin_memory=False
    )

    df = pd.read_csv(RESULTS_CSV)
    if 'fps_cpu' not in df.columns:
        df['fps_cpu'] = np.nan

    # --- Filter skenario sesuai pilihan ---
    if SCENARIOS_FILTER == "required_only":
        mask = (
            (df['scenario'] == 'Teacher (VideoMAE)') |
            (df['scenario'] == 'Student (MobileViT) no KD') |
            ((df['scenario'] == 'Student+KD') & (df['config'] == REQUIRED_KD_CONFIG))
        )
        target_rows = df[mask]
        print(f"Mode: required_only — {len(target_rows)} skenario "
              f"(Teacher, Student baseline, KD wajib proposal T=11/α=0.45)\n")
    else:
        target_rows = df
        print(f"Mode: all — {len(target_rows)} skenario "
              f"(Teacher + Student baseline + 24 kombinasi KD)\n")
        print("Catatan: ini akan makan waktu lebih lama di CPU. "
              "Pantau log per-skenario; aman dihentikan kapan saja "
              "(Ctrl+C) — hasil yang sudah terukur tetap tersimpan "
              "di akhir tiap skenario.\n")

    model = None
    t_start_all = time.time()

    for idx, row in target_rows.iterrows():
        scenario = row['scenario']
        config   = row['config']
        t0_model = time.time()

        if model is not None:
            del model
            gc.collect()

        if scenario == 'Teacher (VideoMAE)':
            ckpt_path = os.path.join(CHECKPOINT_DIR, "teacher_best.pth")
            mode = 'teacher'
            print(f"[{idx+1}/{len(df)}] Teacher (VideoMAE) — model terberat, "
                  f"paling lama di CPU, mohon tunggu...")
            model = build_teacher()

        elif scenario == 'Student (MobileViT) no KD':
            ckpt_path = os.path.join(CHECKPOINT_DIR, "student_baseline_best.pth")
            mode = 'student'
            print(f"[{idx+1}/{len(df)}] Student baseline (no KD, mean) ...")
            model = MobileViTVideo(temporal_type='mean').to(DEVICE)

        elif scenario == 'Student+KD':
            temporal  = row['temporal']
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"student_kd_{config}_best.pth")
            mode = 'student'
            tag = " [WAJIB PROPOSAL]" if config == REQUIRED_KD_CONFIG else ""
            print(f"[{idx+1}/{len(df)}] KD {config} (temporal={temporal}){tag} ...")
            model = MobileViTVideo(temporal_type=temporal).to(DEVICE)

        else:
            print(f"[{idx+1}/{len(df)}] Scenario tidak dikenali, skip.")
            model = None
            continue

        if not os.path.exists(ckpt_path):
            print(f"  → SKIP, checkpoint tidak ditemukan: {ckpt_path}")
            model = None
            continue

        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

        fps = measure_fps(model, test_loader, mode=mode)
        df.at[idx, 'fps_cpu'] = round(fps, 2)
        elapsed = time.time() - t0_model
        print(f"  → FPS_CPU = {fps:.2f}  (durasi pengukuran: {elapsed:.1f}s)")

        # Simpan progresif — aman kalau mau dihentikan di tengah jalan
        df.to_csv(RESULTS_CSV, index=False)

        if COOLDOWN_BETWEEN_MODELS > 0:
            time.sleep(COOLDOWN_BETWEEN_MODELS)

    if model is not None:
        del model
        gc.collect()

    total_elapsed = time.time() - t_start_all
    print(f"\nTotal waktu: {total_elapsed/60:.1f} menit")
    print(f"Hasil tersimpan di kolom 'fps_cpu' → {RESULTS_CSV}")

    # -------------------------
    # Ringkasan khusus CPU
    # -------------------------
    with open(SUMMARY_CPU_TXT, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CPU FPS — PROOF OF CONCEPT EDGE DEVICE\n")
        f.write(f"Device: {DEVICE} | Batch: {FPS_BATCH_SIZE} | Workers: {FPS_WORKERS}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Scenario':<42} {'FPS (GPU)':>10} {'FPS (CPU)':>10}\n")
        f.write("-" * 70 + "\n")
        for _, r in df.iterrows():
            if pd.notna(r.get('fps_cpu')):
                label = f"{r['scenario']} {r['config']}"
                fps_gpu_str = f"{float(r['fps']):.2f}" if pd.notna(r.get('fps')) else "-"
                f.write(f"{label:<42} {fps_gpu_str:>10} {float(r['fps_cpu']):>10.2f}\n")

    print(f"Ringkasan CPU vs GPU → {SUMMARY_CPU_TXT}")
    print("\nSelesai!")