import math, os, glob, random, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.amp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import (distance_transform_edt,
                           binary_erosion, binary_dilation,
                           generate_binary_structure)

from data_loader import (GaussianNoise, RescaleT, RandomCrop, CenterCrop,
                         RandomHorizontalFlip, RandomVerticalFlip,
                         ColorJitter, ToTensor, ToTensorLab, SalObjDataset, RandomRotation, GaussianBlur)
# Ưu tiên model lite cho RTX 3050; nếu chưa copy file thì fallback về BASNet cũ.
try:
    from model.BASNet_3050_lite import BASNet
    print("[Model] Using BASNet_3050_lite")
except Exception:
    from model import BASNet
    print("[Model] Using original BASNet")
import pytorch_ssim
import pytorch_iou

# ================================================================
#  OOM / CUDA CONFIG
# ================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled       = True
torch.backends.cudnn.allow_tf32    = True
torch.backends.cuda.matmul.allow_tf32 = True

# ================================================================
#  DEVICE
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM  : {total_vram:.1f} GB")
    torch.cuda.empty_cache()

# ================================================================
#  CONFIG  ← chỉnh theo VRAM của bạn
# ================================================================
#  VRAM  | batch_size_train | SIZE_TRAIN | CROP_SIZE
#  4 GB  |        1         |    192     |    160
#  6 GB  |        1         |    224     |    192
#  8 GB  |        2         |    256     |    224   ← default
#  12 GB |        4         |    256     |    224
#  16 GB |        8         |    256     |    224
accumulation_steps   = 8       # RTX 3050: effective batch = 8, update nhiều hơn accum=32
batch_size_train     = 1      # RTX 3050 4GB → bắt buộc bs=1
batch_size_val       = 1      # RTX 3050 4GB → bắt buộc bs=1
epoch_num            = 20      # SOD nên train lâu hơn một chút, LR cosine sẽ tự giảm
train_num            = None    # None = dùng toàn bộ ảnh còn lại sau val split
val_num              = None    # None = lấy VAL_RATIO từ toàn bộ dataset
SIZE_TRAIN           = 224     # nếu OOM thì hạ xuống 192
CROP_SIZE            = 192     # nếu OOM thì hạ xuống 160
VAL_RATIO            = 0.15    # dùng gần hết DUTS-TR thay vì bỏ phí ảnh
FREEZE_LOW_ENCODER_UNTIL = 10
UNFREEZE_ALL_EPOCH       = 20

# ── FORCE RESTART ───────────────────────────────────────────────
# Đặt True để xóa checkpoint cũ và train lại từ đầu hoàn toàn.
# Sau khi chạy xong lần đầu, đặt lại False để resume bình thường.
FORCE_RESTART = False   # đổi kiến trúc/loss thì nên train lại từ đầu

# ================================================================
#  LOSS CONFIG
# ================================================================
LOSS_W_BCE  = 1.0
LOSS_W_SSIM = 0.8
LOSS_W_IOU  = 1.0
LOSS_W_EDGE = 0.25   # nhấn nhẹ vào biên mask, rẻ hơn boundary loss phức tạp

# ================================================================
#  CHECKPOINT PATH
# ================================================================
CHECKPOINT_PATH = "./saved_models/basnet_bsi/checkpoint_resume.pth"

# ================================================================
#  CHECKPOINT  SAVE / LOAD
# ================================================================
def save_checkpoint(epoch, net, optimizer, scheduler, scaler,
                    best_val_loss, best_epoch,
                    train_losses, val_losses, lr_list,
                    mae_list, sm_list, em_list, wfm_list, bfm_list):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    tmp_path = CHECKPOINT_PATH + ".tmp"
    import shutil
    torch.save({
        "epoch": epoch, "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "model_state":  net.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "sched_state":  scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "train_losses": train_losses, "val_losses": val_losses,
        "lr_list": lr_list, "mae_list": mae_list,
        "sm_list": sm_list, "em_list": em_list,
        "wfm_list": wfm_list, "bfm_list": bfm_list,
    }, tmp_path)
    shutil.move(tmp_path, CHECKPOINT_PATH)
    print(f"  [Checkpoint] Saved epoch {epoch+1} → {CHECKPOINT_PATH}")

def load_checkpoint(net, optimizer, scheduler, scaler):
    if not os.path.exists(CHECKPOINT_PATH):
        print("[Checkpoint] Not found — training from scratch.")
        return 0, float('inf'), 0, [], [], [], [], [], [], [], []
    print(f"[Checkpoint] Found: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    # strict=False: bỏ qua key mismatch giữa BN (checkpoint cũ) và GN (model mới).
    # Weight/bias affine vẫn được load đúng; chỉ bỏ qua running_mean/var/num_batches.
    missing, unexpected = net.load_state_dict(ckpt["model_state"], strict=True)
    print(f"Layers bị thiếu: {missing}")
    print(f"Layers bị thừa: {unexpected}")
    # Lọc ra những key thực sự quan trọng (không phải BN buffer)
    bn_buffers = {"running_mean", "running_var", "num_batches_tracked"}
    real_missing    = [k for k in missing    if not any(b in k for b in bn_buffers)]
    real_unexpected = [k for k in unexpected if not any(b in k for b in bn_buffers)]
    if real_missing:
        print(f"  [WARN] Missing keys   : {real_missing[:5]} ...")
    if real_unexpected:
        print(f"  [WARN] Unexpected keys: {real_unexpected[:5]} ...")
    skipped = len(missing) + len(unexpected) - len(real_missing) - len(real_unexpected)
    if skipped:
        print(f"  [OK] Skipped {skipped} BN buffer keys (BN→GN convert)")

    # Optimizer / scheduler có thể không tương thích nếu LR thay đổi → reset nhẹ
    try:
        optimizer.load_state_dict(ckpt["optim_state"])
    except Exception as e:
        print(f"  [WARN] Optimizer state không load được ({e}), reset optimizer.")

    try:
        scheduler.load_state_dict(ckpt["sched_state"])
    except Exception as e:
        print(f"  [WARN] Scheduler state không load được ({e}), reset scheduler.")

    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])

    se = ckpt["epoch"] + 1
    print(f"  → Resume epoch {se+1}/{epoch_num} | "
          f"best: {ckpt['best_val_loss']:.6f} (ep{ckpt['best_epoch']})")
    return (se, ckpt["best_val_loss"], ckpt["best_epoch"],
            ckpt["train_losses"], ckpt["val_losses"], ckpt["lr_list"],
            ckpt["mae_list"], ckpt["sm_list"], ckpt["em_list"],
            ckpt["wfm_list"], ckpt["bfm_list"])

# ================================================================
#  METRICS
# ================================================================
def compute_mae(pred, gt):
    p = pred.squeeze().cpu().float().numpy().astype(np.float64)
    g = gt.squeeze().cpu().float().numpy().astype(np.float64)
    return float(np.mean(np.abs(p - g)))

def _ssim_region(pred, gt):
    x, y = pred.mean(), gt.mean()
    sx   = ((pred - x) ** 2).mean()
    sy   = ((gt   - y) ** 2).mean()
    sxy  = ((pred - x) * (gt - y)).mean()
    num  = 4 * x * y * sxy
    den  = (x**2 + y**2) * (sx + sy) + 1e-8
    if num == 0:
        return 1.0 if den == 0 else 0.0
    return float(num / den)

def _centroid(gt):
    r, c = gt.shape
    if gt.sum() == 0:
        return c // 2, r // 2
    jj, ii = np.meshgrid(np.arange(c), np.arange(r))
    t = gt.sum()
    return int((jj * gt).sum() / t), int((ii * gt).sum() / t)

def _divide(a, X, Y):
    return a[:Y, :X], a[:Y, X:], a[Y:, :X], a[Y:, X:]

def _So(p, g):
    u = g.mean()
    return (u * _ssim_region(p * g, g)
            + (1 - u) * _ssim_region((1 - p) * (1 - g), 1 - g))

def _Sr(p, g):
    X, Y  = _centroid(g)
    total = g.shape[0] * g.shape[1]
    return sum(gp.size / total * _ssim_region(pp, gp)
               for gp, pp in zip(_divide(g, X, Y), _divide(p, X, Y)))

def compute_smeasure(pred, gt, alpha=0.5):
    p = pred.squeeze().cpu().float().numpy().astype(np.float64)
    g = (gt.squeeze().cpu().float().numpy() >= 0.5).astype(np.float64)
    y = g.mean()
    if   y == 0: s = 1.0 - p.mean()
    elif y == 1: s = p.mean()
    else:        s = alpha * _So(p, g) + (1 - alpha) * _Sr(p, g)
    return float(max(0.0, s))

def compute_emeasure(pred, gt):
    p = pred.squeeze().cpu().float().numpy().astype(np.float64)
    g = (gt.squeeze().cpu().float().numpy() >= 0.5).astype(np.float64)
    if   g.max() == 0: return float((1 - p).mean())
    elif g.min() == 1: return float(p.mean())
    dp, dg = p - p.mean(), g - g.mean()
    align  = 2 * dp * dg / (dp**2 + dg**2 + 1e-8)
    return float(((align + 1) ** 2 / 4).mean())

def compute_wfmeasure(pred, gt, beta_sq=1.0):
    p = pred.squeeze().cpu().float().numpy().astype(np.float64)
    g = (gt.squeeze().cpu().float().numpy() >= 0.5).astype(np.float64)
    if g.max() == 0:
        return 0.0
    Do = distance_transform_edt(1 - g)
    Di = distance_transform_edt(g)
    if Do.max() > 0: Do = Do / Do.max()
    if Di.max() > 0: Di = Di / Di.max()
    w  = np.where(g > 0.5, 1 + 5 * np.abs(Di - 0.5), 1 + 5 * Do)
    TP = (p * g * w).sum()
    FP = (p * (1 - g) * w).sum()
    FN = ((1 - p) * g * w).sum()
    P  = TP / (TP + FP + 1e-8)
    R  = TP / (TP + FN + 1e-8)
    return float((1 + beta_sq) * P * R / (beta_sq * P + R + 1e-8))

def compute_boundary_fmeasure(pred, gt, threshold=0.5, beta_sq=0.3, tolerance=3):
    pb = (pred.squeeze().cpu().float().numpy() >= threshold).astype(bool)
    gb = (gt.squeeze().cpu().float().numpy()   >= 0.5).astype(bool)
    if pb.sum() == 0 or pb.sum() == pb.size: return 0.0
    if gb.sum() == 0 or gb.sum() == gb.size: return 0.0
    s    = generate_binary_structure(2, 1)
    pb_bd = (pb ^ binary_erosion(pb, s)).astype(np.uint8)
    gb_bd = (gb ^ binary_erosion(gb, s)).astype(np.uint8)
    if pb_bd.sum() == 0: pb_bd = pb.astype(np.uint8)
    if gb_bd.sum() == 0: gb_bd = gb.astype(np.uint8)
    pb_d = binary_dilation(pb_bd.astype(bool), iterations=tolerance).astype(np.uint8)
    gb_d = binary_dilation(gb_bd.astype(bool), iterations=tolerance).astype(np.uint8)
    ps, gs = pb_bd.sum(), gb_bd.sum()
    P = float((pb_bd & gb_d).sum()) / (ps + 1e-8) if ps > 0 else 0.0
    R = float((gb_bd & pb_d).sum()) / (gs + 1e-8) if gs > 0 else 0.0
    return float((1 + beta_sq) * P * R / (beta_sq * P + R + 1e-8))

# ================================================================
#  LOSS FUNCTIONS
# ================================================================
bce_loss  = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss  = pytorch_iou.IOU(size_average=True)

def gradient_map(x):
    """Lightweight edge map for boundary-aware supervision."""
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx + dy

def edge_loss(pred, target):
    return F.l1_loss(gradient_map(pred), gradient_map(target))

def bce_ssim_iou_loss(pred, target, use_edge=False):
    loss = (bce_loss(pred, target)           * LOSS_W_BCE
            + (1 - ssim_loss(pred, target))  * LOSS_W_SSIM
            + iou_loss(pred, target)         * LOSS_W_IOU)
    if use_edge:
        loss = loss + edge_loss(pred, target) * LOSS_W_EDGE
    return loss

def bce_iou_loss(pred, target):
    return bce_loss(pred, target) * LOSS_W_BCE + iou_loss(pred, target) * LOSS_W_IOU

def muti_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v):
    # Tập trung mạnh vào d0/d1, giảm trọng số các side output rất sâu.
    # Cách này thường giúp mask cuối sắc hơn và giảm tình trạng over-supervise.
    loss0 = bce_ssim_iou_loss(d0, labels_v, use_edge=True)
    loss1 = bce_ssim_iou_loss(d1, labels_v, use_edge=False)
    loss2 = bce_iou_loss(d2, labels_v)
    loss3 = bce_iou_loss(d3, labels_v)
    loss4 = bce_iou_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)
    loss7 = bce_loss(d7, labels_v)

    total_loss = (1.00 * loss0 + 0.70 * loss1 + 0.50 * loss2 + 0.40 * loss3
                  + 0.30 * loss4 + 0.15 * loss5 + 0.10 * loss6 + 0.10 * loss7)
    return loss0, total_loss

# ================================================================
#  WARMUP + COSINE SCHEDULER
# ================================================================
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup cho warmup_steps bước đầu,
    sau đó cosine annealing từ max_lr → min_lr.
    """
    def __init__(self, optimizer, warmup_steps, total_steps,
                 min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs  = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                # Linear warmup: 0 → base_lr
                lr = base_lr * (step + 1) / self.warmup_steps
            else:
                # Cosine decay: base_lr → min_lr
                progress = min(1.0, (step - self.warmup_steps) / max(
                    1, self.total_steps - self.warmup_steps))
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs

# ================================================================
#  MAIN TRAIN
# ================================================================
def train():

    # ── DATA ────────────────────────────────────────────────────
    data_dir      = './train_data/'
    tra_image_dir = 'DUTS/DUTS-TR/DUTS-TR-Image/'
    tra_label_dir = 'DUTS/DUTS-TR/DUTS-TR-Mask/'
    model_dir     = "./saved_models/basnet_bsi/"

    all_img = glob.glob(data_dir + tra_image_dir + '*.jpg')
    all_lbl = [os.path.join(data_dir, tra_label_dir,
               os.path.splitext(os.path.basename(p))[0] + '.png')
               for p in all_img]
    combined = list(zip(all_img, all_lbl))
    random.seed(42)
    random.shuffle(combined)
    all_img, all_lbl = map(list, zip(*combined))

    # Dùng gần hết dữ liệu thay vì cố định 6500/1625.
    # Nếu muốn train nhanh để test code, có thể đặt train_num/val_num nhỏ lại ở CONFIG.
    if train_num is None or val_num is None:
        real_val_num = max(1, int(len(all_img) * VAL_RATIO))
        real_train_num = len(all_img) - real_val_num
    else:
        real_train_num = min(train_num, len(all_img))
        real_val_num = min(val_num, max(0, len(all_img) - real_train_num))

    train_dataset = SalObjDataset(
        img_name_list=all_img[:real_train_num],
        lbl_name_list=all_lbl[:real_train_num],
        transform=transforms.Compose([
            RescaleT(SIZE_TRAIN),
            RandomCrop(CROP_SIZE),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.1),           # SOD: flip dọc quá nhiều dễ làm dữ liệu kém tự nhiên
            RandomRotation(degrees=10, p=0.4),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, p=0.4),
            GaussianBlur(kernel_size=3, p=0.15), # blur ít thôi để không làm mềm biên
            GaussianNoise(mean=0.0, std=0.008, p=0.25),
            ToTensorLab(flag=0),
        ])
    )
    val_dataset = SalObjDataset(
        img_name_list=all_img[real_train_num:real_train_num + real_val_num],
        lbl_name_list=all_lbl[real_train_num:real_train_num + real_val_num],
        transform=transforms.Compose([
            RescaleT(SIZE_TRAIN),
            ToTensorLab(flag=0),
        ])
    )

    num_workers = 2

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_train,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size_val,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False,
        persistent_workers=(num_workers > 0)
    )

    print(f"\nTrain: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"(bs={batch_size_train} × accum={accumulation_steps})")
    print(f"Input size: {SIZE_TRAIN} → crop {CROP_SIZE}")
    print(f"Loss: BCE×{LOSS_W_BCE} | SSIM×{LOSS_W_SSIM} | IOU×{LOSS_W_IOU}")

    # ── MODEL ───────────────────────────────────────────────────
    net = BASNet(3, 1).to(device)

    # Freeze tầng thấp vài epoch đầu: tiết kiệm VRAM + giữ đặc trưng ImageNet ổn định.
    for param in net.encoder1.parameters(): param.requires_grad = False
    for param in net.encoder2.parameters(): param.requires_grad = False
    # encoder3/4 vẫn train với LR nhỏ để thích nghi với mask SOD.
    # print(net)
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params/1e6:.2f}M")
    print(f"Model on: {next(net.parameters()).device}")

    # ── OPTIMIZER ───────────────────────────────────────────────
    # Differential LR: decoder/refine học nhanh hơn, encoder pretrained học chậm hơn.
    INIT_LR = 1e-4
    ENCODER_LR = 1e-5
    encoder_params, decoder_params = [], []
    for name, p in net.named_parameters():
        if name.startswith(("encoder1", "encoder2", "encoder3", "encoder4")):
            encoder_params.append(p)
        else:
            decoder_params.append(p)

    optimizer = optim.AdamW(
        [
            {"params": decoder_params, "lr": INIT_LR},
            {"params": encoder_params, "lr": ENCODER_LR},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=5e-5
    )

    # ── SCHEDULER ───────────────────────────────────────────────
    total_steps   = epoch_num * len(train_loader) // accumulation_steps
    warmup_steps  = 3 * len(train_loader) // accumulation_steps
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=1e-7
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ── LOAD CHECKPOINT ─────────────────────────────────────────
    if FORCE_RESTART and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("[FORCE_RESTART] Đã xóa checkpoint cũ — train từ đầu.")

    (start_epoch, best_val_loss, best_epoch,
     train_losses, val_losses, lr_list,
     mae_list, sm_list, em_list, wfm_list, bfm_list) = load_checkpoint(
        net, optimizer, scheduler, scaler)
    
    if start_epoch >= FREEZE_LOW_ENCODER_UNTIL:
        for param in net.encoder1.parameters(): param.requires_grad = True
        for param in net.encoder2.parameters(): param.requires_grad = True
        print(">>> [Resume Confirm] Đã mở khóa encoder1/2.")
    if start_epoch >= UNFREEZE_ALL_EPOCH:
        for param in net.parameters():
            param.requires_grad = True
        print(">>> [Resume Confirm] Đã mở khóa toàn bộ mạng cho giai đoạn Fine-tuning!")
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable_params/1e6:.2f}M")
        print(f"Model on: {next(net.parameters()).device}")

    if start_epoch >= epoch_num:
        print(f"Done ({epoch_num} ep). Delete checkpoint to retrain.")
        raise SystemExit

    print(f"\n--- Training from epoch {start_epoch + 1}/{epoch_num} ---")
    print(f"LR: {INIT_LR:.0e}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        free, tot = torch.cuda.mem_get_info()
        print(f"GPU: {free / 1024**3:.1f} GB free / {tot / 1024**3:.1f} GB total")

    best_wfm = max(wfm_list) if wfm_list else 0.0
    best_mae = min(mae_list) if mae_list  else float('inf')

    # ================================================================
    #  TRAINING LOOP
    # ================================================================
    for epoch in range(start_epoch, epoch_num):

        # ── TRAIN ────────────────────────────────────────────────
        net.train()
        optimizer.zero_grad(set_to_none=True)

        if epoch == FREEZE_LOW_ENCODER_UNTIL:
            for param in net.encoder1.parameters(): param.requires_grad = True
            for param in net.encoder2.parameters(): param.requires_grad = True
            print("--- [UNFREEZE] Mở encoder1/2, tiếp tục fine-tune với LR nhỏ ---")

        if epoch == UNFREEZE_ALL_EPOCH:
            for param in net.parameters():
                param.requires_grad = True
            print("--- [UNFREEZE] Bắt đầu tinh chỉnh toàn bộ mạng ---")

        epoch_train_total = running_d0 = running_total = 0.0
        n_iter       = 0
        total_batches = len(train_loader)
        window_size   = 0

        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch+1}/{epoch_num} [train]",
                         leave=True)

        for i, data in enumerate(train_bar):
            n_iter      += 1
            window_size += 1

            is_last    = (i + 1 == total_batches)
            should_step = ((i + 1) % accumulation_steps == 0) or is_last

            inputs = torch.nan_to_num(
                data['image'].float().to(device, non_blocking=True), 0.0)
            labels = torch.clamp(
                torch.nan_to_num(
                    data['label'].float().to(device, non_blocking=True), 0.0),
                0.0, 1.0)

            # autocast chỉ bao forward của model — BCELoss không an toàn với fp16
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs)

            # Cast về float32 TRƯỚC khi tính loss (BCELoss/SSIM/IOU cần fp32)
            d0f = d0.float(); d1f = d1.float()
            d2f = d2.float(); d3f = d3.float()
            d4f = d4.float(); d5f = d5.float()
            d6f = d6.float(); d7f = d7.float()

            loss_d0, loss = muti_loss_fusion(
                d0f, d1f, d2f, d3f, d4f, d5f, d6f, d7f, labels)

            loss_norm = loss / accumulation_steps

            running_d0        += loss_d0.item()
            running_total     += loss.item()
            epoch_train_total += loss.item()

            scaler.scale(loss_norm).backward()

            del d0, d1, d2, d3, d4, d5, d6, d7
            del d0f, d1f, d2f, d3f, d4f, d5f, d6f, d7f
            del loss_d0, loss, loss_norm
            inputs = labels = None

            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                window_size = 0

            if (i + 1) % 5 == 0:
                train_bar.set_postfix({
                    "d0":    f"{running_d0 / n_iter:.4f}",
                    "total": f"{running_total / n_iter:.4f}",
                    "lr":    f"{optimizer.param_groups[0]['lr']:.2e}",
                })

        epoch_train_total /= total_batches
        train_losses.append(epoch_train_total)

        # ── VALIDATION ───────────────────────────────────────────
        net.eval()

        val_loss     = 0.0
        val_iter     = 0
        total_mae = total_sm = total_em = total_wfm = total_bfm = 0.0
        total_samples = 0

        val_bar = tqdm(val_loader,
                       desc=f"Epoch {epoch+1}/{epoch_num} [val]",
                       leave=True)

        with torch.no_grad():
            for i, data in enumerate(val_bar):
                val_iter += 1
                inputs = torch.nan_to_num(
                    data['image'].float().to(device, non_blocking=True), 0.0)
                labels = torch.clamp(
                    torch.nan_to_num(
                        data['label'].float().to(device, non_blocking=True), 0.0),
                    0.0, 1.0)

                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                    d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs)

                d0f = d0.float(); d1f = d1.float()
                d2f = d2.float(); d3f = d3.float()
                d4f = d4.float(); d5f = d5.float()
                d6f = d6.float(); d7f = d7.float()

                _, val_batch = muti_loss_fusion(
                    d0f, d1f, d2f, d3f, d4f, d5f, d6f, d7f, labels)
                val_loss += val_batch.item()

                pred = d0f
                for b in range(pred.shape[0]):
                    total_mae += compute_mae(pred[b], labels[b])
                    total_sm  += compute_smeasure(pred[b], labels[b])
                    total_em  += compute_emeasure(pred[b], labels[b])
                    total_wfm += compute_wfmeasure(pred[b], labels[b])
                    total_bfm += compute_boundary_fmeasure(pred[b], labels[b])
                    total_samples += 1

                del (d0, d1, d2, d3, d4, d5, d6, d7,
                     d0f, d1f, d2f, d3f, d4f, d5f, d6f, d7f,
                     pred, val_batch)

                if (i + 1) % 5 == 0:
                    val_bar.set_postfix({
                        "MAE":  f"{total_mae / total_samples:.4f}",
                        "Sm":   f"{total_sm  / total_samples:.3f}",
                        "Em":   f"{total_em  / total_samples:.3f}",
                        "wFm":  f"{total_wfm / total_samples:.3f}",
                        "bFm":  f"{total_bfm / total_samples:.3f}",
                        "loss": f"{val_loss  / val_iter:.4f}",
                    })

        net.train()

        avg_val_loss = val_loss / val_iter
        avg_mae = total_mae / total_samples
        avg_sm  = total_sm  / total_samples
        avg_em  = total_em  / total_samples
        avg_wfm = total_wfm / total_samples
        avg_bfm = total_bfm / total_samples

        val_losses.append(avg_val_loss)
        mae_list.append(avg_mae);  sm_list.append(avg_sm)
        em_list.append(avg_em);    wfm_list.append(avg_wfm)
        bfm_list.append(avg_bfm)

        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)

        print(f"\n[Epoch {epoch+1}] "
              f"Train(total)={epoch_train_total:.4f} | "
              f"Val(total)={avg_val_loss:.4f}")
        print(f"  wFm={avg_wfm:.4f}  bFm={avg_bfm:.4f}  "
              f"MAE={avg_mae:.4f}(↓)  Sm={avg_sm:.4f}  Em={avg_em:.4f}")
        print(f"  LR → {current_lr:.2e}")

        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GB used | "
                  f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved")

        # ── SAVE BEST ────────────────────────────────────────────
        os.makedirs(model_dir, exist_ok=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch    = epoch + 1
            torch.save(net.state_dict(),
                       model_dir + "basnet_best_valloss.pth")
            print("  ✓ Saved best val loss")

        if len(sm_list) == 1 or avg_sm >= max(sm_list[:-1]):
            torch.save(net.state_dict(),
                       model_dir + "basnet_best_sm.pth")
            print("  ✓ Saved best Sm")

        if len(wfm_list) == 1 or avg_wfm >= max(wfm_list[:-1]):
            best_wfm = avg_wfm
            torch.save(net.state_dict(),
                       model_dir + "basnet_best_wfm.pth")
            print("  ✓ Saved best WFm")

        if avg_mae < best_mae:
            best_mae = avg_mae
            torch.save(net.state_dict(),
                       model_dir + "basnet_best_mae.pth")
            print("  ✓ Saved best MAE")

        print(f"  Best → loss={best_val_loss:.6f}(ep{best_epoch}) "
              f"WFm={best_wfm:.4f} MAE={best_mae:.4f}\n")

        save_checkpoint(epoch, net, optimizer, scheduler, scaler,
                        best_val_loss, best_epoch,
                        train_losses, val_losses, lr_list,
                        mae_list, sm_list, em_list, wfm_list, bfm_list)

        # ── PLOTS ────────────────────────────────────────────────
        os.makedirs("figures", exist_ok=True)
        x = list(range(1, len(train_losses) + 1))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, train_losses, label="Train (8 outputs)",
                color='#1f77b4', marker='o', markersize=3, lw=2)
        ax.plot(x, val_losses,   label="Val (8 outputs)",
                color='#ff7f0e', marker='o', markersize=3, lw=2)
        if best_epoch > 0:
            ax.axvline(x=best_epoch, ls='--', color='red',
                       label=f'Best({best_epoch})')
            ax.scatter(best_epoch, val_losses[best_epoch - 1],
                       color='red', zorder=5, s=50)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3); ax.legend()
        plt.title(f"Loss Curve — Best ep={best_epoch}")
        plt.tight_layout()
        plt.savefig("figures/loss_curve.png", dpi=150)
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for vals, lbl, mk in [(wfm_list, "wFm", 'o'),
                               (bfm_list, "bFm", 's'),
                               (sm_list,  "Sm",  '^'),
                               (em_list,  "Em",  'd')]:
            axes[0].plot(x, vals, label=lbl, lw=2, marker=mk, markersize=3)
        axes[0].set_ylim(0, 1); axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="lower right", fontsize=9)
        axes[0].set_title("Segmentation Metrics (↑)")
        axes[1].plot(x, mae_list, color='tomato', lw=2,
                     marker='o', markersize=3)
        axes[1].grid(True, alpha=0.3); axes[1].set_title("MAE (↓)")
        plt.suptitle(
            f"BCE×{LOSS_W_BCE}|SSIM×{LOSS_W_SSIM}|IOU×{LOSS_W_IOU} | "
            f"size={SIZE_TRAIN}→crop{CROP_SIZE}",
            fontsize=11)
        plt.tight_layout()
        plt.savefig("figures/metrics_curve.png", dpi=150, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, lr_list, lw=2, marker='o', markersize=3)
        ax.set_yscale('log'); ax.grid(True)
        plt.title("LR Schedule (warmup + cosine)")
        plt.tight_layout()
        plt.savefig("figures/lr_schedule.png", dpi=150)
        plt.close()

    print(f"\nBest epoch : {best_epoch} | Val loss: {best_val_loss:.6f}")
    print(f"Best WFm   : {best_wfm:.4f} | Best MAE: {best_mae:.4f}")
    print("--- Training Done! ---")

if __name__ == '__main__':
    train()
