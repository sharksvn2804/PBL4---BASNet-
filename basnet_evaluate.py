# ============================================================
#  DUTS-TE EVALUATION SCRIPT
#  Run this AFTER training to evaluate on test set
#  Usage: python basnet_evaluate.py
# ============================================================

import torch
import os
import glob
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.amp
from tqdm import tqdm
from scipy.ndimage import (
    distance_transform_edt,
    binary_erosion,
    binary_dilation,
    generate_binary_structure,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from data_loader import (
    RescaleT, ToTensorLab, SalObjDataset, ToTensor
)
from model import BASNet

# ============================================================
#  CONFIG
# ============================================================

# ── Số ảnh muốn evaluate (đặt None để dùng toàn bộ tập) ──
NUM_SAMPLES = 1000
RANDOM_SEED = 42          # seed để tái lặp kết quả

FIGURES_DIR = "./figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
#  EVALUATION METRICS (same as basnet_train.py - DO NOT CHANGE)
# ============================================================

def compute_mae(pred, gt):
    pred_np = pred.squeeze().cpu().float().numpy().astype(np.float64)
    gt_np   = gt.squeeze().cpu().float().numpy().astype(np.float64)
    return float(np.mean(np.abs(pred_np - gt_np)))

def _ssim_region(pred, gt):
    x = pred.mean()
    y = gt.mean()
    sigma_x  = ((pred - x) ** 2).mean()
    sigma_y  = ((gt  - y) ** 2).mean()
    sigma_xy = ((pred - x) * (gt - y)).mean()
    num = 4 * x * y * sigma_xy
    den = (x ** 2 + y ** 2) * (sigma_x + sigma_y) + 1e-8
    if num == 0:
        return 1.0 if den == 0 else 0.0
    return float(num / den)

def _centroid(gt):
    rows, cols = gt.shape
    if gt.sum() == 0:
        return cols // 2, rows // 2
    j_grid, i_grid = np.meshgrid(np.arange(cols), np.arange(rows))
    total = gt.sum()
    X = int((j_grid * gt).sum() / total)
    Y = int((i_grid * gt).sum() / total)
    return X, Y

def _divide(arr, X, Y):
    return arr[:Y, :X], arr[:Y, X:], arr[Y:, :X], arr[Y:, X:]

def _So(pred, gt):
    fg = pred * gt
    bg = (1 - pred) * (1 - gt)
    u  = gt.mean()
    return u * _ssim_region(fg, gt) + (1 - u) * _ssim_region(bg, 1 - gt)

def _Sr(pred, gt):
    X, Y  = _centroid(gt)
    h, w  = gt.shape
    total = h * w
    score = 0.0
    for gp, pp in zip(_divide(gt, X, Y), _divide(pred, X, Y)):
        w_i = gp.size / total
        score += w_i * _ssim_region(pp, gp)
    return score

def compute_smeasure(pred, gt, alpha=0.5):
    pred_np = pred.squeeze().cpu().float().numpy().astype(np.float64)
    gt_np   = (gt.squeeze().cpu().float().numpy() >= 0.5).astype(np.float64)
    y = gt_np.mean()
    if y == 0:
        score = 1.0 - pred_np.mean()
    elif y == 1:
        score = pred_np.mean()
    else:
        score = alpha * _So(pred_np, gt_np) + (1 - alpha) * _Sr(pred_np, gt_np)
    return float(max(0.0, score))

def compute_emeasure(pred, gt):
    pred_np = pred.squeeze().cpu().float().numpy()
    gt_np = (gt.squeeze().cpu().float().numpy() >= 0.5).astype(np.float64)
    
    th = 2 * pred_np.mean()
    if th > 1: th = 1
    pred_bin = (pred_np >= th).astype(np.float64)
    
    if gt_np.max() == 0:
        enhanced = 1.0 - pred_bin
    elif gt_np.min() == 1:
        enhanced = pred_bin
    else:
        mu_p = pred_bin.mean()
        mu_g = gt_np.mean()
        dp = pred_bin - mu_p
        dg = gt_np - mu_g
        align = 2.0 * dp * dg / (dp ** 2 + dg ** 2 + 1e-8)
        enhanced = ((align + 1.0) ** 2) / 4.0
    
    return float(enhanced.mean())

def compute_wfmeasure(pred, gt, beta_sq=1.0):
    pred_np = pred.squeeze().cpu().float().numpy().astype(np.float64)
    gt_np   = (gt.squeeze().cpu().float().numpy() >= 0.5).astype(np.float64)

    if gt_np.max() == 0:
        return 0.0

    E = np.abs(pred_np - gt_np)
    Dst = distance_transform_edt(1 - gt_np)
    if Dst.max() > 0:
        Dst = Dst / Dst.max()

    weight = 1.0 + 5.0 * Dst
    TP = (gt_np * (1 - E) * weight).sum()
    FP = ((1 - gt_np) * E * weight).sum()

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (gt_np.sum() + 1e-8)

    Fw = (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-8)
    return float(Fw)

def compute_boundary_fmeasure(pred, gt, threshold=0.5, beta_sq=0.3, tolerance=3):
    pred_bin = (pred.squeeze().cpu().float().numpy() >= threshold).astype(np.uint8)
    gt_bin   = (gt.squeeze().cpu().float().numpy()   >= 0.5      ).astype(np.uint8)

    struct      = generate_binary_structure(2, 1)
    pred_bd     = pred_bin ^ binary_erosion(pred_bin, struct).astype(np.uint8)
    gt_bd       = gt_bin   ^ binary_erosion(gt_bin,   struct).astype(np.uint8)

    pred_bd_dil = binary_dilation(pred_bd, iterations=tolerance)
    gt_bd_dil   = binary_dilation(gt_bd,   iterations=tolerance)

    pred_bd_sum = pred_bd.sum()
    gt_bd_sum   = gt_bd.sum()

    precision = float((pred_bd & gt_bd_dil).sum()) / (pred_bd_sum + 1e-8) \
                if pred_bd_sum > 0 else 0.0
    recall    = float((gt_bd & pred_bd_dil).sum()) / (gt_bd_sum + 1e-8) \
                if gt_bd_sum > 0 else 0.0

    Fb = (1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-8)
    return float(Fb)


# ============================================================
#  PLOTTING HELPERS
# ============================================================

def _style_ax(ax, title, ylabel, color):
    """Áp dụng style chung cho một subplot."""
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel("Sample index", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_four_metrics(wfm_list, bfm_list, sm_list, em_list, save_path):
    """
    Vẽ 4 metrics (wFm, bFm, Sm, Em) trên 1 figure (2×2 grid).
    Mỗi panel có: scatter per-sample + running-average line + final-avg hline.
    """
    metrics = [
        (wfm_list, "Weighted F-measure  (F^w)", "F^w",  "#4C72B0"),
        (bfm_list, "Boundary F-measure  (F^b)", "F^b",  "#DD8452"),
        (sm_list,  "S-measure  (S_α)",           "S_α",  "#55A868"),
        (em_list,  "E-measure  (E_m)",            "E_m",  "#C44E52"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"BASNet — Per-sample Metrics on DUTS-TE  (n={len(wfm_list)})",
        fontsize=15, fontweight='bold', y=1.01
    )
    axes = axes.flatten()

    for ax, (values, title, ylabel, color) in zip(axes, metrics):
        xs = np.arange(1, len(values) + 1)
        arr = np.array(values)
        running_avg = np.cumsum(arr) / xs

        ax.scatter(xs, arr, s=6, alpha=0.35, color=color, label="Per-sample")
        ax.plot(xs, running_avg, lw=2.0, color=color, label="Running avg")

        final_avg = arr.mean()
        ax.axhline(final_avg, color='black', lw=1.3, linestyle='--',
                   label=f"Final avg = {final_avg:.4f}")

        _style_ax(ax, title, ylabel, color)
        ax.legend(fontsize=8.5, loc='lower right')
        ax.set_ylim(max(0, arr.min() - 0.05), min(1.05, arr.max() + 0.05))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Saved → {save_path}")


def plot_mae(mae_list, save_path):
    """
    Vẽ riêng biểu đồ MAE: scatter + running-average line + final-avg hline.
    """
    arr = np.array(mae_list)
    xs  = np.arange(1, len(arr) + 1)
    running_avg = np.cumsum(arr) / xs
    final_avg   = arr.mean()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(xs, arr, s=6, alpha=0.35, color="#8172B2", label="Per-sample MAE")
    ax.plot(xs, running_avg, lw=2.0, color="#8172B2", label="Running avg")
    ax.axhline(final_avg, color='black', lw=1.3, linestyle='--',
               label=f"Final avg = {final_avg:.4f}")

    _style_ax(ax, f"MAE — DUTS-TE  (n={len(arr)})", "MAE  (↓ lower is better)", "#8172B2")
    ax.legend(fontsize=9)
    ax.set_ylim(max(0, arr.min() - 0.01), arr.max() + 0.01)

    fig.suptitle("BASNet — Mean Absolute Error on DUTS-TE",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Saved → {save_path}")


# ============================================================
#  MAIN EVALUATION
# ============================================================

if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    # ------- Model Loading --------
    model_path = "./saved_models/basnet_bsi/basnet_best_mae.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        print("Please train the model first using basnet_train.py")
        exit(1)
    
    net = BASNet(3, 1).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    print(f"[LOAD] Model loaded from {model_path}")
    print(f"[DEVICE] {device}")
    
    # ------- DUTS-TE Dataset --------
    test_data_dir = './validation_data/'
    test_image_dir = 'DUTS-TE/DUTS-TE-Image/'
    test_label_dir = 'DUTS-TE/DUTS-TE-Mask/'
    image_ext = '.jpg'
    label_ext = '.png'
    
    SIZE_TEST = 224
    
    all_img_paths = sorted(
        glob.glob(test_data_dir + test_image_dir + '*' + image_ext)
    )

    # ── Lấy subset ngẫu nhiên (có seed để tái lặp) ──────────────
    if NUM_SAMPLES is not None and NUM_SAMPLES < len(all_img_paths):
        random.seed(RANDOM_SEED)
        test_img_name_list = random.sample(all_img_paths, NUM_SAMPLES)
        test_img_name_list.sort()          # sắp lại để dễ debug
        print(f"[DATA] Subset: {NUM_SAMPLES}/{len(all_img_paths)} images "
              f"(seed={RANDOM_SEED})")
    else:
        test_img_name_list = all_img_paths
        print(f"[DATA] Using full test set: {len(test_img_name_list)} images")
    
    test_lbl_name_list = []
    for img_path in test_img_name_list:
        img_name = os.path.basename(img_path)
        imidx = os.path.splitext(img_name)[0]
        test_lbl_name_list.append(
            os.path.join(test_data_dir, test_label_dir, imidx + label_ext)
        )
    
    # ------- DataLoader --------
    test_dataset = SalObjDataset(
        img_name_list=test_img_name_list,
        lbl_name_list=test_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(SIZE_TEST),
            ToTensorLab(flag=0)
        ])
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    # ------- Evaluation --------
    print("\n[EVAL] Evaluating on DUTS-TE subset...")

    # Per-sample lists (dùng để vẽ đồ thị)
    mae_list = []
    sm_list  = []
    em_list  = []
    wfm_list = []
    bfm_list = []

    total_samples = 0
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="DUTS-TE Evaluation", leave=True)
        for data in test_bar:
            inputs, labels = data['image'], data['label']
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                d0, _, _, _, _, _, _, _ = net(inputs)
            
            pred = d0.float()
            
            for b in range(pred.shape[0]):
                mae_val = compute_mae(pred[b], labels[b])
                sm_val  = compute_smeasure(pred[b], labels[b])
                em_val  = compute_emeasure(pred[b], labels[b])
                wfm_val = compute_wfmeasure(pred[b], labels[b])
                bfm_val = compute_boundary_fmeasure(pred[b], labels[b])

                mae_list.append(mae_val)
                sm_list.append(sm_val)
                em_list.append(em_val)
                wfm_list.append(wfm_val)
                bfm_list.append(bfm_val)
                total_samples += 1
            
            test_bar.set_postfix({
                "MAE": f"{np.mean(mae_list):.4f}",
                "Sm":  f"{np.mean(sm_list):.3f}",
                "Em":  f"{np.mean(em_list):.3f}",
                "wFm": f"{np.mean(wfm_list):.3f}",
                "bFm": f"{np.mean(bfm_list):.3f}",
            })
            
            torch.cuda.empty_cache()
    
    # ------- Results --------
    avg_mae = np.mean(mae_list)
    avg_sm  = np.mean(sm_list)
    avg_em  = np.mean(em_list)
    avg_wfm = np.mean(wfm_list)
    avg_bfm = np.mean(bfm_list)
    
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS ON DUTS-TE")
    print("="*60)
    print(f"  (1) Weighted Fm  F^w : {avg_wfm:.4f}  -- Best")
    print(f"  (2) Boundary Fm  F^b : {avg_bfm:.4f}  -- Best")
    print(f"  (3) MAE          M   : {avg_mae:.4f}  (down, lower better)")
    print(f"  (4) S-measure    S_a : {avg_sm:.4f}  -- Best")
    print(f"  (5) E-measure    E_m : {avg_em:.4f}  -- Best")
    print("="*60)
    print(f"Total samples: {total_samples}")
    print("="*60 + "\n")

    # ------- Plots --------
    print("[PLOT] Generating figures...")

    plot_four_metrics(
        wfm_list, bfm_list, sm_list, em_list,
        save_path=os.path.join(FIGURES_DIR, "metrics_wfm_bfm_sm_em.png")
    )

    plot_mae(
        mae_list,
        save_path=os.path.join(FIGURES_DIR, "metric_mae.png")
    )

    print(f"\n[DONE] All figures saved to '{FIGURES_DIR}/'")