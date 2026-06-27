# ============================================================
# PURE COMPUTE BENCHMARK — Forward Pass Murni, Tanpa Decode
# ============================================================
# Latar belakang: FPS end-to-end (pipeline lengkap: decode video + model)
# menunjukkan Teacher (94.2M parameter) dan Student (~5.7M parameter)
# punya throughput yang HAMPIR SAMA — baik di GPU maupun lebih ekstrem
# lagi di CPU (rasio cuma ~1.08x, padahal beda parameter 16.5x).
#
# Hipotesis: overhead decode video (OpenCV, CPU-bound, SAMA untuk semua
# model) mendominasi waktu total, sehingga perbedaan kompleksitas
# komputasi model jadi nyaris tidak kelihatan di pengukuran end-to-end.
#
# Script ini membuktikan/membantah hipotesis itu dengan cara:
#   1. Ambil SATU batch tensor yang SUDAH di-decode & ditransformasi
#      (sekali saja, di awal script).
#   2. Forward pass model BERULANG-ULANG di tensor yang SAMA itu —
#      tidak ada decode baru, tidak ada I/O disk sama sekali di loop
#      yang diukur waktunya.
#   3. Hasilnya murni mencerminkan biaya komputasi model itu sendiri.
#
# Kalau hipotesis benar: Teacher akan jauh lebih lambat dari Student
# di sini (mendekati rasio 16.5x), beda jauh dari hasil end-to-end yang
# cuma 1.08x. Itu pembuktian definitif kenapa FPS end-to-end gap-nya
# kecil — bukan karena modelnya sama cepat, tapi karena pipeline-nya
# yang jadi bottleneck dominan.
#
# Tidak menyentuh kolom 'fps' atau 'fps_cpu' yang sudah ada — menambah
# kolom baru 'fps_pure_gpu' dan 'fps_pure_cpu'.
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
NUM_FRAMES     = 16
NUM_CLASSES    = 5
TEACHER_SIZE   = 224
STUDENT_SIZE   = 256

BENCH_BATCH_SIZE = 4
N_WARM           = 10
N_RUN            = 50   # bisa lebih banyak dari versi pipeline — tidak ada I/O wait

TEST_CSV = "/home/coder/data_skripsi/dataset_gabungan_siap_training/test_metadata.csv"
CHECKPOINT_DIR = "/home/coder/output_model/skenario_raffi/checkpoints"
RESULTS_DIR    = "/home/coder/output_model/skenario_raffi/results"
RESULTS_CSV    = os.path.join(RESULTS_DIR, "experiment_results.csv")
SUMMARY_PURE_TXT = os.path.join(RESULTS_DIR, "summary_pure_compute.txt")

# -------------------------
# Label & transform — identik dengan script sebelumnya
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
# Model definitions — identik
# -------------------------
def build_teacher():
    # map_location/posisi device ditentukan belakangan, jangan auto .to(device) di sini
    m = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    return m

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
# Pure compute measurement — TIDAK ada decode/I-O di sini sama sekali
# -------------------------
def measure_pure_fps(model, tb_cached, ts_cached, mode, device,
                      n_warm=N_WARM, n_run=N_RUN):
    model.eval()
    model.to(device)
    tb = tb_cached.to(device)
    ts = ts_cached.to(device)

    def _fwd():
        return model(pixel_values=tb) if mode == 'teacher' else model(ts)

    with torch.no_grad():
        for _ in range(n_warm):
            _fwd()

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_run):
            _fwd()
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    total_frames = tb.shape[0] * tb.shape[1] * n_run
    return total_frames / elapsed

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    gpu_available = torch.cuda.is_available()
    print(f"GPU tersedia: {gpu_available}\n")

    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Tidak ditemukan: {RESULTS_CSV}")

    print("Mengambil SATU batch tensor (decode SEKALI saja, dipakai berulang)...")
    loader = DataLoader(
        FPSDataset(TEST_CSV), batch_size=BENCH_BATCH_SIZE,
        shuffle=True, num_workers=2
    )
    tb_cached, ts_cached, _ = next(iter(loader))
    print(f"  Batch Teacher : {tuple(tb_cached.shape)}")
    print(f"  Batch Student : {tuple(ts_cached.shape)}\n")

    df = pd.read_csv(RESULTS_CSV)
    if 'fps_pure_gpu' not in df.columns:
        df['fps_pure_gpu'] = np.nan
    if 'fps_pure_cpu' not in df.columns:
        df['fps_pure_cpu'] = np.nan

    model = None
    t_start = time.time()

    for idx, row in df.iterrows():
        scenario = row['scenario']
        config   = row['config']

        if model is not None:
            del model
            if gpu_available:
                torch.cuda.empty_cache()
            gc.collect()

        if scenario == 'Teacher (VideoMAE)':
            ckpt_path = os.path.join(CHECKPOINT_DIR, "teacher_best.pth")
            mode = 'teacher'
            print(f"[{idx+1}/{len(df)}] Teacher (VideoMAE) ...")
            model = build_teacher()

        elif scenario == 'Student (MobileViT) no KD':
            ckpt_path = os.path.join(CHECKPOINT_DIR, "student_baseline_best.pth")
            mode = 'student'
            print(f"[{idx+1}/{len(df)}] Student baseline (no KD) ...")
            model = MobileViTVideo(temporal_type='mean')

        elif scenario == 'Student+KD':
            temporal  = row['temporal']
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"student_kd_{config}_best.pth")
            mode = 'student'
            print(f"[{idx+1}/{len(df)}] KD {config} (temporal={temporal}) ...")
            model = MobileViTVideo(temporal_type=temporal)

        else:
            print(f"[{idx+1}/{len(df)}] Scenario tidak dikenali, skip.")
            model = None
            continue

        if not os.path.exists(ckpt_path):
            print(f"  → SKIP, checkpoint tidak ditemukan: {ckpt_path}")
            model = None
            continue

        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

        if gpu_available:
            fps_gpu = measure_pure_fps(model, tb_cached, ts_cached, mode, 'cuda')
            df.at[idx, 'fps_pure_gpu'] = round(fps_gpu, 2)
            print(f"  → Pure GPU compute : {fps_gpu:.2f} FPS")

        fps_cpu = measure_pure_fps(model, tb_cached, ts_cached, mode, 'cpu')
        df.at[idx, 'fps_pure_cpu'] = round(fps_cpu, 2)
        print(f"  → Pure CPU compute : {fps_cpu:.2f} FPS")

        df.to_csv(RESULTS_CSV, index=False)

    total_elapsed = time.time() - t_start
    print(f"\nTotal waktu benchmark: {total_elapsed:.1f}s")
    print(f"Hasil tersimpan di kolom 'fps_pure_gpu' & 'fps_pure_cpu' → {RESULTS_CSV}")

    # -------------------------
    # Ringkasan perbandingan: end-to-end vs pure compute
    # -------------------------
    with open(SUMMARY_PURE_TXT, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("PERBANDINGAN: FPS END-TO-END (pipeline+model) vs PURE COMPUTE (model saja)\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"{'Scenario':<40} {'FPS GPU':>9} {'FPS CPU':>9} "
                f"{'Pure GPU':>9} {'Pure CPU':>9}\n")
        f.write("-" * 90 + "\n")
        for _, r in df.iterrows():
            label = f"{r['scenario']} {r['config']}"
            fg  = f"{float(r['fps']):.2f}"       if pd.notna(r.get('fps'))           else "-"
            fc  = f"{float(r['fps_cpu']):.2f}"   if pd.notna(r.get('fps_cpu'))        else "-"
            pg  = f"{float(r['fps_pure_gpu']):.2f}" if pd.notna(r.get('fps_pure_gpu')) else "-"
            pc  = f"{float(r['fps_pure_cpu']):.2f}" if pd.notna(r.get('fps_pure_cpu')) else "-"
            f.write(f"{label:<40} {fg:>9} {fc:>9} {pg:>9} {pc:>9}\n")

        # Hitung rasio Teacher vs rata-rata Student di tiap kolom — bukti hipotesis
        teacher_row = df[df['scenario'] == 'Teacher (VideoMAE)'].iloc[0]
        student_rows = df[df['scenario'] != 'Teacher (VideoMAE)']

        f.write("\n\nRASIO TEACHER vs RATA-RATA STUDENT-FAMILY (parameter: 94.2M vs ~5.7M, 16.5x)\n")
        f.write("-" * 90 + "\n")
        for col, label in [('fps', 'End-to-end GPU'), ('fps_cpu', 'End-to-end CPU'),
                            ('fps_pure_gpu', 'Pure compute GPU'), ('fps_pure_cpu', 'Pure compute CPU')]:
            if pd.notna(teacher_row.get(col)) and student_rows[col].notna().any():
                t_val = float(teacher_row[col])
                s_avg = float(student_rows[col].dropna().astype(float).mean())
                ratio = s_avg / t_val if t_val > 0 else float('nan')
                f.write(f"{label:<20} Teacher={t_val:>8.2f}  Student_avg={s_avg:>8.2f}  "
                        f"Rasio={ratio:.2f}x\n")

    print(f"Ringkasan perbandingan → {SUMMARY_PURE_TXT}")
    print("\nSelesai! Cek summary_pure_compute.txt — lihat bagian RASIO untuk")
    print("bukti langsung apakah bottleneck-nya pipeline atau model compute.")