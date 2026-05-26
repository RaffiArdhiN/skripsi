# ============================================================
# EXPERIMENT SWEEP — Sekali run, semua skenario
# Skenario yang dijalankan:
#   1. Teacher (VideoMAE) baseline — early stopping + max 800 epoch
#   2. Student (MobileViT) tanpa KD — mean pooling baseline
#   3. Student + KD — grid search alpha × T × temporal_type
#
# Perubahan dari versi sebelumnya:
#   - Early stopping teacher & student (patience=30) — aman secara akademis,
#     800 tetap jadi batas atas sesuai proposal Tabel 3.2
#   - alpha=1.0 tetap sebagai eksplorasi tambahan (pure KD loss)
#   - LSTM sebagai eksplorasi variasi temporal aggregation
#   - Bug fix: variabel T di results.append diganti kd_temp
#   - Epoch disesuaikan lebih realistis untuk data ~320 sampel
#
# Output: results/experiment_results.csv + results/summary.txt
# ============================================================
# Layer-wise LR menggantikan single LR=1.5e-4 dari proposal
# untuk mencegah destruksi pretrained weights — justifikasi: Howard & Ruder (2018)
# import os
import os, time, random, itertools
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import matplotlib
matplotlib.use('Agg')  # wajib di server tanpa GUI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
import cv2
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from transformers import VideoMAEForVideoClassification

# -------------------------
# Config dasar
# -------------------------
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES     = 16
TEACHER_BATCH  = 8    # VideoMAE lebih berat — 8 aman di 32GB VRAM
KD_BATCH = 4
STUDENT_BATCH  = 4   # MobileViT ringan — bisa lebih besar
NUM_WORKERS    = 4
SEED           = 42
NUM_CLASSES    = 5
TEACHER_SIZE   = 224
STUDENT_SIZE   = 256

TRAIN_CSV = "/home/coder/data_skripsi/dataset_gabungan_siap_training/train_metadata.csv"
VAL_CSV   = "/home/coder/data_skripsi/dataset_gabungan_siap_training/val_metadata.csv"
TEST_CSV  = "/home/coder/data_skripsi/dataset_gabungan_siap_training/test_metadata.csv"

CHECKPOINT_DIR = "/home/coder/output_model/skenario_raffi/checkpoints"
RESULTS_DIR    = "/home/coder/output_model/skenario_raffi/results"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Desain eksperimen
# -------------------------
# Skenario 1: Teacher — maks 800 epoch sesuai proposal Tabel 3.2,
#   dengan early stopping patience=30 agar tidak overfit pada data kecil
TEACHER_EPOCHS  = 800
TEACHER_PATIENCE = 30

# Skenario 2: Student tanpa KD — maks 300 epoch sesuai proposal Tabel 3.3
STUDENT_EPOCHS   = 300
STUDENT_PATIENCE = 30
LR_STUDENT       = 2e-3

# Skenario 3: Grid search KD
#   alpha  : default 0.45 (Ahmadabadi et al., 2023); range eksplorasi simetris ±0.15
#             alpha=1.0 sebagai eksplorasi tambahan (pure KD loss tanpa CE)
#   T      : default 11 (Ahmadabadi et al., 2023); range eksplorasi mengacu
#             Hinton et al. (2015) bahwa T>8 stabil untuk model besar
#   temporal: mean (baseline proposal) vs lstm (eksplorasi variasi agregasi temporal,
#             Donahue et al., 2015)
KD_ALPHA_LIST  = [0.3, 0.45, 0.6, 1.0]
KD_T_LIST      = [7.0, 11.0, 15.0]
KD_EPOCHS      = 300
KD_PATIENCE    = 30
LR_KD          = 1e-4
TEMPORAL_TYPES = ['mean', 'lstm']

# -------------------------
# Verifikasi & stratified split
# -------------------------
def verify_split_distribution():
    print("=" * 50)
    print("Verifikasi distribusi label per split:")
    for name, path in [("TRAIN", TRAIN_CSV), ("VAL", VAL_CSV), ("TEST", TEST_CSV)]:
        df = pd.read_csv(path)
        print(f"\n{name} ({len(df)} samples):")
        print(df['label'].value_counts().to_string())
    print("=" * 50)
    print("Cek apakah distribusi antar split sudah proporsional!")
    print("Kalau belum, gunakan make_stratified_split() di bawah.\n")

def make_stratified_split(all_csv_path, train_ratio=0.8, seed=SEED):
    """
    Gunakan fungsi ini kalau CSV kamu belum di-split secara stratified.
    all_csv_path: path ke CSV yang berisi SEMUA data (sebelum di-split)
    """
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(all_csv_path)
    train_df, temp_df = train_test_split(
        df, train_size=train_ratio, stratify=df['label'], random_state=seed)
    val_df, test_df = train_test_split(
        temp_df, train_size=0.5, stratify=temp_df['label'], random_state=seed)

    out_train = os.path.join(CHECKPOINT_DIR, "train_stratified.csv")
    out_val   = os.path.join(CHECKPOINT_DIR, "val_stratified.csv")
    out_test  = os.path.join(CHECKPOINT_DIR, "test_stratified.csv")
    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val,     index=False)
    test_df.to_csv(out_test,   index=False)

    print(f"Stratified split selesai:")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"  Disimpan di: {CHECKPOINT_DIR}")
    return out_train, out_val, out_test

# -------------------------
# Label mapping
# -------------------------
label_to_idx = {
    "1_mengangguk": 0,
    "2_mengangkat_tangan": 1,
    "3_menggunakan_hp": 2,
    "4_menopang_kepala": 3,
    "5_menunduk": 4
}
idx_to_label = {v: k for k, v in label_to_idx.items()}

# -------------------------
# Transforms
# -------------------------
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

def make_transforms(size, augment=False):
    if augment:
        return T.Compose([
            T.ToPILImage(),
            T.Resize((size, size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResizedCrop((size, size), scale=(0.9, 1.0)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    return T.Compose([
        T.ToPILImage(),
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

# -------------------------
# Dataset
# -------------------------
class CSVDatasetCV(Dataset):
    def __init__(self, csv_path, num_frames=NUM_FRAMES, split='train', augment=False):
        self.df         = pd.read_csv(csv_path)
        self.num_frames = num_frames
        self.split      = split
        self.tf_teacher = make_transforms(TEACHER_SIZE, augment=(augment and split=='train'))
        self.tf_student = make_transforms(STUDENT_SIZE, augment=(augment and split=='train'))

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
                               dtype=np.uint8)] * self.num_frames

        total   = len(frames)
        indices = np.linspace(0, total-1, self.num_frames).astype(int) \
                  if total >= self.num_frames \
                  else np.pad(np.linspace(0, total-1, total).astype(int),
                              (0, self.num_frames - total), mode='wrap')
        sampled = [frames[i] for i in indices]

        t_teach = torch.stack([self.tf_teacher(f) for f in sampled])
        t_stud  = torch.stack([self.tf_student(f)  for f in sampled])
        return t_teach, t_stud, label

def make_loaders(augment=False, batch_size=STUDENT_BATCH):
    kw = dict(batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=True)
    return (
        DataLoader(CSVDatasetCV(TRAIN_CSV, split='train', augment=augment),
                   shuffle=True,  **kw),
        DataLoader(CSVDatasetCV(VAL_CSV,   split='val',   augment=False),
                   shuffle=False, **kw),
        DataLoader(CSVDatasetCV(TEST_CSV,  split='test',  augment=False),
                   shuffle=False, **kw),
    )

# -------------------------
# Models
# -------------------------
def build_teacher():
    m = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True  # classifier head beda ukuran, ini expected
    )
    return m.to(DEVICE)

class MobileViTVideo(nn.Module):
    """
    MobileViT dengan dua opsi agregasi temporal:
      - 'mean' : temporal average pooling (baseline proposal)
      - 'lstm' : LSTM agregasi sekuens frame (eksplorasi, Donahue et al., 2015)
    """
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
        B, T, C, H, W = x.shape
        feats = self.backbone(x.view(B*T, C, H, W)).view(B, T, -1)
        if self.temporal_type == 'lstm':
            _, (h_n, _) = self.temporal_layer(feats)
            pooled = h_n[-1]
        else:
            pooled = feats.mean(1)
        return self.classifier(pooled)

# -------------------------
# KD Loss
# -------------------------
class KDLoss(nn.Module):
    def __init__(self, alpha, T):
        super().__init__()
        self.alpha = alpha
        self.T     = T
        self.ce    = nn.CrossEntropyLoss()
        self.kl    = nn.KLDivLoss(reduction='batchmean')

    def forward(self, stu, tea, labels):
        kd = self.kl(F.log_softmax(stu/self.T, 1),
                     F.softmax(tea/self.T, 1)) * self.T**2
        # alpha=1.0: pure KD loss (eksplorasi), CE diabaikan
        return self.alpha * kd + (1 - self.alpha) * self.ce(stu, labels)

# -------------------------
# Train 1 epoch
# -------------------------
def train_one_epoch_teacher(model, loader, optimizer):
    model.train()
    total = 0.0
    for tb, _, labels in loader:
        tb, labels = tb.to(DEVICE), labels.to(DEVICE)
        loss = F.cross_entropy(model(pixel_values=tb).logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item()
    return total / len(loader)

# def train_one_epoch_student(model, loader, optimizer, criterion):
#     model.train()
#     total = 0.0
#     for _, ts, labels in loader:
#         ts, labels = ts.to(DEVICE), labels.to(DEVICE)
#         loss = criterion(model(ts), labels)
#         optimizer.zero_grad(); loss.backward(); optimizer.step()
#         total += loss.item()
#     return total / len(loader)

def train_one_epoch_student(model, loader, optimizer, criterion, accum_steps=4):
    model.train()
    total = 0.0
    optimizer.zero_grad()
    for i, (_, ts, labels) in enumerate(loader):
        ts, labels = ts.to(DEVICE), labels.to(DEVICE)
        loss = criterion(model(ts), labels) / accum_steps
        loss.backward()
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
        total += loss.item() * accum_steps
    return total / len(loader)

# def train_one_epoch_kd(student, teacher_model, loader, optimizer, criterion):
#     student.train()
#     total = 0.0
#     for tb, ts, labels in loader:
#         tb, ts, labels = tb.to(DEVICE), ts.to(DEVICE), labels.to(DEVICE)
#         with torch.no_grad():
#             tea_logits = teacher_model(pixel_values=tb).logits
#         loss = criterion(student(ts), tea_logits, labels)
#         optimizer.zero_grad(); loss.backward(); optimizer.step()
#         total += loss.item()
#     return total / len(loader)

def train_one_epoch_kd(student, teacher_model, loader, optimizer, criterion, accum_steps=4):
    student.train()
    total = 0.0
    optimizer.zero_grad()
    for i, (tb, ts, labels) in enumerate(loader):
        tb, ts, labels = tb.to(DEVICE), ts.to(DEVICE), labels.to(DEVICE)
        with torch.no_grad():
            tea_logits = teacher_model(pixel_values=tb).logits
        loss = criterion(student(ts), tea_logits, labels) / accum_steps
        loss.backward()
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
        total += loss.item() * accum_steps
    return total / len(loader)

# -------------------------
# Evaluate + FPS
# -------------------------
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, loader, mode, scenario_name):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for tb, ts, labels in loader:
            labels = labels.to(DEVICE)
            logits = model(pixel_values=tb.to(DEVICE)).logits \
                     if mode == 'teacher' else model(ts.to(DEVICE))
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(labels.cpu().tolist())
    
    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=[idx_to_label[i] for i in range(NUM_CLASSES)],
                yticklabels=[idx_to_label[i] for i in range(NUM_CLASSES)])
    plt.title(f'Confusion Matrix — {scenario_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'cm_{scenario_name}.png'), dpi=150)
    plt.close()

def evaluate(model, loader, mode='student', scenario_name=None):
    """
    Evaluasi model dan kembalikan metrik.
    Kalau scenario_name diberikan, confusion matrix
    otomatis disimpan ke RESULTS_DIR/cm_{scenario_name}.png
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for tb, ts, labels in loader:
            labels = labels.to(DEVICE)
            logits = model(pixel_values=tb.to(DEVICE)).logits \
                     if mode == 'teacher' else model(ts.to(DEVICE))
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(labels.cpu().tolist())
 
    acc = accuracy_score(trues, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        trues, preds, average='macro', zero_division=0)
    report = classification_report(
        trues, preds,
        target_names=[idx_to_label[i] for i in range(NUM_CLASSES)],
        zero_division=0)
 
    # Confusion matrix — hanya kalau scenario_name diberikan
    if scenario_name is not None:
        cm = confusion_matrix(trues, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', ax=ax,
            xticklabels=[idx_to_label[i] for i in range(NUM_CLASSES)],
            yticklabels=[idx_to_label[i] for i in range(NUM_CLASSES)],
        )
        ax.set_title(f'Confusion Matrix — {scenario_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f'cm_{scenario_name}.png')
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  → Confusion matrix disimpan: cm_{scenario_name}.png")
 
    return acc, p, r, f1, report

def measure_fps(model, loader, mode='student', n_warm=5, n_run=20):
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

# -------------------------
# Collector hasil
# -------------------------
results = []

def save_results():
    pd.DataFrame(results).to_csv(
        os.path.join(RESULTS_DIR, "experiment_results.csv"), index=False)
    print(f"Results saved → {os.path.join(RESULTS_DIR, 'experiment_results.csv')}")

    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            f.write(f"[{r['scenario']}]\n")
            f.write(f"  Config    : {r['config']}\n")
            f.write(f"  Stopped   : ep {r['stopped_epoch']}\n")
            f.write(f"  Test Acc  : {r['test_acc']:.4f}  "
                    f"P: {r['precision']:.4f}  "
                    f"R: {r['recall']:.4f}  "
                    f"F1: {r['f1']:.4f}\n")
            f.write(f"  FPS       : {r['fps']:.2f} frames/s\n")
            f.write(f"  Val Acc   : {r['val_acc']:.4f}\n\n")

        f.write("\nTRADE-OFF TABLE (Test Acc vs FPS)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Scenario':<40} {'Acc':>6} {'F1':>6} {'FPS':>8}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['scenario']+' '+r['config']:<40} "
                    f"{r['test_acc']:>6.4f} {r['f1']:>6.4f} {r['fps']:>8.2f}\n")
    print(f"Summary saved  → {summary_path}")

# ============================================================
# LANGKAH 0: Verifikasi distribusi split
# ============================================================
verify_split_distribution()
SKIP_TO_SCENARIO = 2

if SKIP_TO_SCENARIO <= 1:
    # ============================================================
    # SKENARIO 1: Teacher (VideoMAE)
    # ============================================================
    print("\n" + "="*60)
    print("SKENARIO 1: Teacher (VideoMAE) baseline")
    print("="*60)

    teacher_model = build_teacher()
    # optimizer_t   = torch.optim.AdamW(
    #     teacher_model.parameters(), lr=1.5e-4, weight_decay=0.05)
    # scheduler_t   = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_t, T_max=TEACHER_EPOCHS, eta_min=1e-6)
    optimizer_t = torch.optim.AdamW([
        {'params': teacher_model.videomae.parameters(), 'lr': 5e-6},   # encoder
        {'params': teacher_model.classifier.parameters(), 'lr': 1e-4}  # head baru
    ], weight_decay=0.05)
    scheduler_t = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_t, mode='max', factor=0.5, patience=10)

    train_loader, val_loader, test_loader = make_loaders(
        augment=True, batch_size=TEACHER_BATCH) 

    best_val_f1_t    = 0.0
    no_improve_t     = 0
    stopped_epoch_t  = TEACHER_EPOCHS

    for ep in range(TEACHER_EPOCHS):
        loss = train_one_epoch_teacher(teacher_model, train_loader, optimizer_t)
    
        # evaluate DULU — val_f1_ep baru terdefinisi di sini
        val_acc_ep, _, _, val_f1_ep, _ = evaluate(
            teacher_model, val_loader, mode='teacher')
    
        # scheduler.step SETELAH evaluate — pakai val_f1_ep yang sudah ada
        scheduler_t.step(val_f1_ep)
    
        # Tidak ada lagi scheduler_t.get_last_lr() karena ReduceLROnPlateau
        # tidak punya method itu — pakai optimizer langsung
        current_lr = optimizer_t.param_groups[0]['lr']
        print(f"  Teacher ep {ep+1}/{TEACHER_EPOCHS} | "
            f"loss {loss:.4f} | val_acc {val_acc_ep:.4f} | "
            f"val_f1 {val_f1_ep:.4f} | lr {current_lr:.2e}")
    
        if val_f1_ep > best_val_f1_t:
            best_val_f1_t = val_f1_ep
            no_improve_t  = 0
            torch.save(teacher_model.state_dict(),
                    os.path.join(CHECKPOINT_DIR, "teacher_best.pth"))
            print(f"  → Best teacher checkpoint disimpan (val_f1={val_f1_ep:.4f})")
        else:
            no_improve_t += 1
            if no_improve_t >= TEACHER_PATIENCE:
                stopped_epoch_t = ep + 1
                print(f"  → Early stopping teacher di epoch {stopped_epoch_t}")
                break
            
    torch.save(teacher_model.state_dict(),
            os.path.join(CHECKPOINT_DIR, "teacher_final.pth"))
    print(f"  → Final teacher checkpoint disimpan")

    # Load best untuk evaluasi final
    teacher_model.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, "teacher_best.pth")))
    
    val_acc_t,  _, _, _,    _   = evaluate(
        teacher_model, val_loader,  mode='teacher')
    test_acc_t, tp, tr, tf1, rp = evaluate(
        teacher_model, test_loader, mode='teacher',
        scenario_name='Teacher_VideoMAE')
    fps_t = measure_fps(teacher_model, test_loader, mode='teacher')

    print(f"\n  Teacher | val_acc={val_acc_t:.4f} | "
        f"test_acc={test_acc_t:.4f} | FPS={fps_t:.2f}")
    print(rp)

    results.append({
        'scenario'     : 'Teacher (VideoMAE)',
        'config'       : f'ep_max={TEACHER_EPOCHS}',
        'alpha'        : '-',
        'T'            : '-',
        'temporal'     : '-',
        'stopped_epoch': stopped_epoch_t,
        'val_acc'      : val_acc_t,
        'test_acc'     : test_acc_t,
        'precision'    : tp,
        'recall'       : tr,
        'f1'           : tf1,
        'fps'          : fps_t,
    })
    save_results()

else:
    # print("Membebaskan VRAM teacher...")
    # teacher_state_dict_path = os.path.join(CHECKPOINT_DIR, "teacher_best.pth")
    # del teacher_model
    # torch.cuda.empty_cache()
    # import gc; gc.collect()

    # Load teacher dari checkpoint yang sudah ada
    print("Skipping Skenario 1 — load dari checkpoint...")
    teacher_model = build_teacher()
    teacher_model.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR, "teacher_best.pth")))
    teacher_model.eval()
    # Isi hasil dari CSV yang sudah ada
    existing = pd.read_csv(os.path.join(RESULTS_DIR, "experiment_results.csv"))
    for _, row in existing.iterrows():
        results.append(row.to_dict())
    test_acc_t = existing[existing['scenario']=='Teacher (VideoMAE)']['test_acc'].values[0]
    fps_t      = existing[existing['scenario']=='Teacher (VideoMAE)']['fps'].values[0]
    print(f"  Teacher loaded | test_acc={test_acc_t:.4f} | FPS={fps_t:.2f}")
print("Membebaskan VRAM teacher sebelum melatih student...")
del teacher_model
torch.cuda.empty_cache()
import gc; gc.collect()
# ============================================================
# SKENARIO 2: Student (MobileViT) tanpa KD — mean pooling baseline
# ============================================================
print("\n" + "="*60)
print("SKENARIO 2: Student (MobileViT) tanpa KD — mean pooling")
print("="*60)

student_base  = MobileViTVideo(temporal_type='mean').to(DEVICE)
optimizer_sb  = torch.optim.AdamW(
    student_base.parameters(), lr=LR_STUDENT, weight_decay=0.01)
scheduler_sb  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_sb, T_max=STUDENT_EPOCHS, eta_min=1e-6)
ce_criterion  = nn.CrossEntropyLoss()

train_loader, val_loader, test_loader = make_loaders(
    augment=True, batch_size=STUDENT_BATCH)

best_val_f1_sb   = 0.0
no_improve_sb    = 0
stopped_epoch_sb = STUDENT_EPOCHS

for ep in range(STUDENT_EPOCHS):
    loss = train_one_epoch_student(
        student_base, train_loader, optimizer_sb, ce_criterion)
    scheduler_sb.step()
    val_acc_ep, _, _, val_f1_ep, _ = evaluate(student_base, val_loader, mode='student')
    print(f"  Student baseline ep {ep+1}/{STUDENT_EPOCHS} | "
          f"loss {loss:.4f} | val_acc {val_acc_ep:.4f} | val_f1 {val_f1_ep:.4f}")

    if val_f1_ep > best_val_f1_sb:
        best_val_f1_sb   = val_f1_ep
        no_improve_sb    = 0
        torch.save(student_base.state_dict(),
                   os.path.join(CHECKPOINT_DIR, "student_baseline_best.pth"))
        print(f"  → Best student baseline checkpoint disimpan "
              f"(val_f1={val_f1_ep:.4f})")
    else:
        no_improve_sb += 1
        if no_improve_sb >= STUDENT_PATIENCE:
            stopped_epoch_sb = ep + 1
            print(f"  → Early stopping student baseline di epoch {stopped_epoch_sb}")
            break

torch.save(student_base.state_dict(),
           os.path.join(CHECKPOINT_DIR, "student_baseline_final.pth"))

student_base.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "student_baseline_best.pth")))

val_acc_sb,  _, _, _,    _  = evaluate(student_base, val_loader,  mode='student')
test_acc_sb, sp, sr, sf1, rp = evaluate(
    student_base, test_loader, mode='student',
    scenario_name='Student_MobileViT_noKD') 
fps_sb = measure_fps(student_base, test_loader, mode='student')

print(f"\n  Student baseline | val_acc={val_acc_sb:.4f} | "
      f"test_acc={test_acc_sb:.4f} | FPS={fps_sb:.2f}")
print(rp)

results.append({
    'scenario'     : 'Student (MobileViT) no KD',
    'config'       : f'ep_max={STUDENT_EPOCHS}_mean',
    'alpha'        : '-',
    'T'            : '-',
    'temporal'     : 'mean',
    'stopped_epoch': stopped_epoch_sb,
    'val_acc'      : val_acc_sb,
    'test_acc'     : test_acc_sb,
    'precision'    : sp,
    'recall'       : sr,
    'f1'           : sf1,
    'fps'          : fps_sb,
})
save_results()

print("Reload teacher untuk KD grid search...")
teacher_model = build_teacher()
teacher_model.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "teacher_best.pth"),
               map_location=DEVICE))
teacher_model.eval()
for p in teacher_model.parameters():
    p.requires_grad = False

# ============================================================
# SKENARIO 3: Grid search KD
# Total: 2 temporal × 4 alpha × 3 T = 24 kombinasi
# ============================================================
total_kd = len(TEMPORAL_TYPES) * len(KD_ALPHA_LIST) * len(KD_T_LIST)
print("\n" + "="*60)
print(f"SKENARIO 3: KD Grid Search — "
      f"{len(TEMPORAL_TYPES)} temporal × {len(KD_ALPHA_LIST)} alpha × "
      f"{len(KD_T_LIST)} T = {total_kd} kombinasi")
print("="*60)

# Freeze teacher — tidak perlu dilatih lagi
teacher_model.eval()
for p in teacher_model.parameters():
    p.requires_grad = False

best_kd = {'f1': 0, 'config': ''}
combo_num = 0

for t_type, alpha, kd_temp in itertools.product(
        TEMPORAL_TYPES, KD_ALPHA_LIST, KD_T_LIST):

    combo_num += 1

    # Nama skenario yang deskriptif
    if alpha == 1.0:
        scen_tag = "PureKD"       # eksplorasi: pure KD tanpa CE
    else:
        scen_tag = "StandardKD"
    if t_type == 'lstm':
        scen_tag += "_LRCN"       # eksplorasi: LSTM temporal

    config_name = f"{scen_tag}_a={alpha}_T={kd_temp}"
    print(f"\n[{combo_num}/{total_kd}] {config_name}")

    student_kd   = MobileViTVideo(temporal_type=t_type).to(DEVICE)
    optimizer_kd = torch.optim.AdamW(
        student_kd.parameters(), lr=LR_KD, weight_decay=0.01)
    scheduler_kd = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_kd, T_max=KD_EPOCHS, eta_min=1e-7)
    kd_criterion = KDLoss(alpha=alpha, T=kd_temp)

    train_loader, val_loader, test_loader = make_loaders(
        augment=True, batch_size=KD_BATCH)

    best_val_f1_kd   = 0.0
    no_improve_kd    = 0
    stopped_epoch_kd = KD_EPOCHS

    for ep in range(KD_EPOCHS):
        loss = train_one_epoch_kd(
            student_kd, teacher_model, train_loader, optimizer_kd, kd_criterion)
        scheduler_kd.step()
        val_acc_ep, _, _, val_f1_ep, _ = evaluate(student_kd, val_loader, mode='student')
        print(f"  ep {ep+1}/{KD_EPOCHS} | loss {loss:.4f} | "
              f"val_acc {val_acc_ep:.4f} | val_f1 {val_f1_ep:.4f}")

        if val_f1_ep > best_val_f1_kd:
            best_val_f1_kd   = val_f1_ep
            no_improve_kd    = 0
            torch.save(student_kd.state_dict(),
                       os.path.join(CHECKPOINT_DIR,
                                    f"student_kd_{config_name}_best.pth"))
            print(f"  → Best checkpoint disimpan (val_f1={val_f1_ep:.4f})")
        else:
            no_improve_kd += 1
            if no_improve_kd >= KD_PATIENCE:
                stopped_epoch_kd = ep + 1
                print(f"  → Early stopping di epoch {stopped_epoch_kd}")
                break

    torch.save(student_kd.state_dict(),
               os.path.join(CHECKPOINT_DIR,
                            f"student_kd_{config_name}_final.pth"))

    # Evaluasi dengan best checkpoint
    student_kd.load_state_dict(
        torch.load(os.path.join(CHECKPOINT_DIR,
                                f"student_kd_{config_name}_best.pth")))

    val_acc_kd,  _,  _,   _,   _  = evaluate(student_kd, val_loader,  mode='student')
    test_acc_kd, kp, kr, kf1, rp = evaluate(
        student_kd, test_loader, mode='student',
        scenario_name=config_name)
    fps_kd = measure_fps(student_kd, test_loader, mode='student')

    print(f"  val_acc={val_acc_kd:.4f} | test_acc={test_acc_kd:.4f} | "
          f"F1={kf1:.4f} | FPS={fps_kd:.2f}")
    print(rp)

    results.append({
        'scenario'     : 'Student+KD',
        'config'       : config_name,
        'alpha'        : alpha,
        'T'            : kd_temp,       # BUG FIX: sebelumnya pakai 'T' (undefined)
        'temporal'     : t_type,
        'stopped_epoch': stopped_epoch_kd,
        'val_acc'      : val_acc_kd,
        'test_acc'     : test_acc_kd,
        'precision'    : kp,
        'recall'       : kr,
        'f1'           : kf1,
        'fps'          : fps_kd,
    })

    if kf1 > best_kd['f1']:
        best_kd = {
            'f1'    : kf1,
            'config': config_name,
            'alpha' : alpha,
            'T'     : kd_temp,
            'acc'   : test_acc_kd,
            'fps'   : fps_kd,
        }

    # Simpan setiap kombinasi selesai — aman kalau koneksi putus
    save_results()

# ============================================================
# RINGKASAN AKHIR
# ============================================================
print("\n" + "="*60)
print("RINGKASAN AKHIR SEMUA SKENARIO")
print("="*60)

df_res = pd.DataFrame(results)
print(df_res[['scenario','config','stopped_epoch',
              'test_acc','f1','fps']].to_string(index=False))

print(f"\nKonfigurasi KD terbaik  : {best_kd['config']}")
print(f"  Test Acc  : {best_kd['acc']:.4f}")
print(f"  F1 macro  : {best_kd['f1']:.4f}")
print(f"  FPS       : {best_kd['fps']:.2f}")
print(f"\nTeacher    | test_acc={test_acc_t:.4f} | FPS={fps_t:.2f}")
print(f"Student    | test_acc={test_acc_sb:.4f} | FPS={fps_sb:.2f}")
print(f"\nSpeedup KD vs Teacher  : {best_kd['fps']/fps_t:.2f}x")
print(f"Acc gap KD vs Teacher  : {test_acc_t - best_kd['acc']:.4f}")

save_results()
print("\nSelesai! Cek folder results/ untuk CSV dan summary lengkap.")