# ============================================================
# UKUR FPS SEMUA SKENARIO — Sekali jalan, kondisi konsisten
# ============================================================
# KENAPA DIPISAH dari training loop:
#
# 1. Konsistensi kondisi sistem. Saat training, FPS tiap kombinasi
#    diukur di jam yang berbeda-beda (training 1 kombinasi bisa 3-4 jam).
#    Kondisi GPU (suhu, beban user lain di lab) ikut berubah antar
#    pengukuran. Setelah hang & restart server (2x), pola FPS yang
#    terukur berubah drastis di tengah grid search — kombinasi sebelum
#    hang ~100-135 FPS, sesudahnya ~165-184 FPS — padahal tidak ada
#    alasan arsitektural untuk itu (LSTM bahkan konsisten "lebih cepat"
#    dari mean pooling, yang harusnya sebaliknya karena LSTM memproses
#    16 timestep secara sekuensial). Itu sinyal kuat FPS yang lama
#    TIDAK comparable satu sama lain.
#
# 2. Batch size yang dulu tidak seragam. Saat training, Teacher diukur
#    dengan TEACHER_BATCH=8 sedangkan Student & semua KD diukur dengan
#    KD_BATCH=4. Batch lebih besar = utilisasi GPU lebih baik = FPS
#    lebih tinggi, independen dari arsitektur model. Script ini pakai
#    SATU batch size yang sama untuk semua 26 skenario.
#
# Script ini TIDAK melatih apa-apa. Hanya load checkpoint yang sudah
# ada, ukur FPS, lalu update kolom 'fps' di experiment_results.csv.
# Kolom lain (test_acc, f1, precision, recall, dst) TIDAK disentuh,
# karena metrik itu tidak bergantung pada kondisi sistem/timing.
# ============================================================

import os, time, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
# Config — SAMA untuk semua pengukuran
# -------------------------
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES     = 16
NUM_CLASSES    = 5
TEACHER_SIZE   = 224
STUDENT_SIZE   = 256

FPS_BATCH_SIZE = 4    # SATU batch size untuk SEMUA model — ini yang dulu beda
FPS_WORKERS    = 4    # konsisten untuk semua, tidak berubah antar pengukuran
N_WARM         = 5
N_RUN          = 20
COOLDOWN_BETWEEN_MODELS = 3   # detik, jeda kecil antar pengukuran

TEST_CSV = "/home/coder/data_skripsi/dataset_gabungan_siap_training/test_metadata.csv"
CHECKPOINT_DIR = "/home/coder/output_model/skenario_raffi/checkpoints"
RESULTS_DIR    = "/home/coder/output_model/skenario_raffi/results"
RESULTS_CSV    = os.path.join(RESULTS_DIR, "experiment_results.csv")
SUMMARY_TXT    = os.path.join(RESULTS_DIR, "summary.txt")

# -------------------------
# Label & transform (identik dengan train_classifier.py)
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

# -------------------------
# Dataset — sama logikanya dengan CSVDatasetCV, disederhanakan
# (cuma butuh test split, tidak perlu augmentasi)
# -------------------------
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
# Model definitions — HARUS identik dengan train_classifier.py
# supaya state_dict checkpoint bisa di-load tanpa key mismatch
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
# FPS measurement — identik logikanya dengan train_classifier.py
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

    if DEVICE == 'cuda': torch.cuda.synchronize()
    t0, total_frames = time.time(), 0
    with torch.no_grad():
        for _ in range(n_run):
            tb, ts, _ = _next()
            _fwd(tb, ts)
            total_frames += tb.shape[0] * tb.shape[1]
    if DEVICE == 'cuda': torch.cuda.synchronize()

    return total_frames / (time.time() - t0)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Tidak ditemukan: {RESULTS_CSV}")

    print("Membuat test loader (dipakai SAMA untuk semua 26 pengukuran)...")
    test_loader = DataLoader(
        FPSDataset(TEST_CSV),
        batch_size=FPS_BATCH_SIZE,
        shuffle=False,
        num_workers=FPS_WORKERS,
        pin_memory=True
    )

    df = pd.read_csv(RESULTS_CSV)
    print(f"Memuat {len(df)} baris dari {RESULTS_CSV}\n")
    print(f"Konfigurasi pengukuran: batch_size={FPS_BATCH_SIZE}, "
          f"num_workers={FPS_WORKERS}, n_warm={N_WARM}, n_run={N_RUN}\n")

    model = None  # tracker untuk cleanup eksplisit antar model

    for idx, row in df.iterrows():
        scenario = row['scenario']
        config   = row['config']

        # --- Bersihkan model SEBELUMNYA sebelum load yang baru ---
        if model is not None:
            del model
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
            print(f"[{idx+1}/{len(df)}] Student baseline (no KD, mean pooling) ...")
            model = MobileViTVideo(temporal_type='mean').to(DEVICE)

        elif scenario == 'Student+KD':
            temporal  = row['temporal']
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"student_kd_{config}_best.pth")
            mode = 'student'
            print(f"[{idx+1}/{len(df)}] KD {config} (temporal={temporal}) ...")
            model = MobileViTVideo(temporal_type=temporal).to(DEVICE)

        else:
            print(f"[{idx+1}/{len(df)}] Scenario tidak dikenali: '{scenario}', skip.")
            model = None
            continue

        if not os.path.exists(ckpt_path):
            print(f"  → SKIP, checkpoint tidak ditemukan: {ckpt_path}")
            model = None
            continue

        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

        fps = measure_fps(model, test_loader, mode=mode)
        df.at[idx, 'fps'] = round(fps, 2)
        print(f"  → FPS = {fps:.2f}")

        if COOLDOWN_BETWEEN_MODELS > 0:
            time.sleep(COOLDOWN_BETWEEN_MODELS)

    # Cleanup model terakhir
    if model is not None:
        del model
        torch.cuda.empty_cache()
        gc.collect()

    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nFPS untuk semua skenario sudah diupdate → {RESULTS_CSV}")

    # -------------------------
    # Regenerate summary.txt dengan FPS yang sudah konsisten
    # -------------------------
    with open(SUMMARY_TXT, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT SUMMARY (FPS diukur ulang — kondisi seragam)\n")
        f.write("=" * 70 + "\n\n")
        for _, r in df.iterrows():
            f.write(f"[{r['scenario']}]\n")
            f.write(f"  Config    : {r['config']}\n")
            f.write(f"  Stopped   : ep {r['stopped_epoch']}\n")
            f.write(f"  Test Acc  : {float(r['test_acc']):.4f}  "
                    f"P: {float(r['precision']):.4f}  "
                    f"R: {float(r['recall']):.4f}  "
                    f"F1: {float(r['f1']):.4f}\n")
            f.write(f"  FPS       : {float(r['fps']):.2f} frames/s\n")
            f.write(f"  Val Acc   : {float(r['val_acc']):.4f}\n\n")

        f.write("\nTRADE-OFF TABLE (Test Acc vs FPS) — batch size & worker SERAGAM\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Scenario':<42} {'Acc':>6} {'F1':>6} {'FPS':>8}\n")
        f.write("-" * 70 + "\n")
        for _, r in df.iterrows():
            label = f"{r['scenario']} {r['config']}"
            f.write(f"{label:<42} "
                    f"{float(r['test_acc']):>6.4f} {float(r['f1']):>6.4f} "
                    f"{float(r['fps']):>8.2f}\n")

    print(f"Summary diperbarui → {SUMMARY_TXT}")
    print("\nSelesai! Semua 26 FPS sekarang diukur dalam kondisi yang sama persis.")