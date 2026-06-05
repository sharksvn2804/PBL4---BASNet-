import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import BASNet


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi)


def predict_single(net, image_path, device):
    """Trả về (ảnh gốc ndarray, mask dự đoán ndarray uint8)."""
    dataset = SalObjDataset(
        img_name_list=[image_path],
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(224), ToTensorLab(flag=0)])
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        for data in loader:
            inputs = data['image'].type(torch.FloatTensor).to(device)
            d1, d2, d3, d4, d5, d6, d7, d8 = net(inputs)
            pred = normPRED(d1[:, 0, :, :])
            del d2, d3, d4, d5, d6, d7, d8

    pred_np = pred.squeeze().cpu().numpy()          # float [0, 1]
    mask    = Image.fromarray((pred_np * 255).astype(np.uint8))
    orig    = io.imread(image_path)
    mask    = mask.resize((orig.shape[1], orig.shape[0]), resample=Image.NEAREST)
    return orig, np.array(mask)


def load_model(model_path, device):
    net = BASNet(3, 1)
    net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    net.to(device)
    net.eval()
    return net


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

IMAGE_DIR   = './test_data/test_images/'
FIGURES_DIR = './figures/'
MODEL_DIR   = './saved_models/basnet_bsi/'

MODELS = {
    'MAE'     : 'basnet_best_mae.pth',
    'SM'      : 'basnet_best_sm.pth',
    'ValLoss' : 'basnet_best_valloss.pth',
    'WFM'     : 'basnet_best_wfm.pth',
}

NUM_SAMPLES = 3
SEED        = None        # None → random khác nhau mỗi lần chạy

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(FIGURES_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1. Chọn ngẫu nhiên ảnh test
    all_images = sorted(
        glob.glob(os.path.join(IMAGE_DIR, '*.jpg')) +
        glob.glob(os.path.join(IMAGE_DIR, '*.png'))
    )
    if not all_images:
        raise FileNotFoundError(f'Không tìm thấy ảnh trong {IMAGE_DIR}')

    rng      = random.Random(SEED)
    selected = rng.sample(all_images, min(NUM_SAMPLES, len(all_images)))
    print('Ảnh được chọn:', [os.path.basename(p) for p in selected])

    # 2. Load tất cả model
    print('Đang load models...')
    nets = {
        name: load_model(os.path.join(MODEL_DIR, fname), device)
        for name, fname in MODELS.items()
    }
    model_names = list(nets.keys())   # ['MAE', 'SM', 'ValLoss', 'WFM']

    # 3. Subplot: rows = ảnh, cols = Original | MAE | SM | ValLoss | WFM
    col_labels = ['Original'] + model_names
    n_rows = len(selected)
    n_cols = len(col_labels)

    fig = plt.figure(figsize=(n_cols * 2.8, n_rows * 2.8))
    gs  = gridspec.GridSpec(n_rows, n_cols, hspace=0.06, wspace=0.04)

    for row, img_path in enumerate(selected):
        stem    = os.path.splitext(os.path.basename(img_path))[0]
        orig_np = io.imread(img_path)

        # Dự đoán với từng model
        preds = {}
        for name, net in nets.items():
            _, mask = predict_single(net, img_path, device)
            preds[name] = mask
            print(f'  [{stem}] model={name} ✓')

        # ── vẽ từng ô ──
        for col, label in enumerate(col_labels):
            ax = fig.add_subplot(gs[row, col])
            ax.set_xticks([]); ax.set_yticks([])

            if label == 'Original':
                ax.imshow(orig_np)
            else:
                ax.imshow(preds[label], cmap='gray', vmin=0, vmax=255)

            # Tiêu đề cột (hàng đầu tiên)
            if row == 0:
                ax.set_title(label, fontsize=10, fontweight='bold', pad=5)

            # Nhãn hàng (cột đầu tiên)
            if col == 0:
                ax.set_ylabel(stem, fontsize=8, labelpad=4)

            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_edgecolor('#555')

    fig.suptitle('BASNet — Kết quả dự đoán theo từng model', fontsize=13, fontweight='bold', y=1.02)

    save_path = os.path.join(FIGURES_DIR, 'model_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure đã lưu → {save_path}')