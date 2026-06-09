Link chứa model (sau fine tune): https://drive.google.com/drive/folders/1fxyhRmLCfPovoN-WosHzqXmPUYlw66bS?usp=drive_link

# BASNet - Salient Object Detection

Repository này dùng để huấn luyện, fine-tune, đánh giá và chạy dự đoán với mô hình **BASNet** cho bài toán phát hiện/phân đoạn vật thể nổi bật (*Salient Object Detection*).

Do dữ liệu và checkpoint mô hình có dung lượng lớn, GitHub chỉ lưu **mã nguồn chính**. Người dùng cần tải thêm **dataset** và **model `.pth`** rồi đặt đúng thư mục như hướng dẫn bên dưới.

---

## 1. Cấu trúc thư mục cần có

Sau khi clone project về máy, cấu trúc thư mục nên được giữ như sau:

```text
.
├── demo/                         # Ảnh/kết quả minh họa
├── figures/                      # Hình ảnh, biểu đồ dùng trong báo cáo
├── model/                        # Code kiến trúc BASNet, không đặt checkpoint .pth ở đây
├── pytorch_iou/                  # Module IoU loss/metric
├── pytorch_ssim/                 # Module SSIM loss/metric
├── saved_models/
│   └── basnet_bsi/               # Đặt các file checkpoint .pth tại đây
│       ├── basnet_best_mae.pth
│       ├── basnet_best_sm.pth
│       ├── basnet_best_valloss.pth
│       ├── basnet_best_wfm.pth
│       ├── basnet_epoch_best.pth
│       ├── checkpoint_finetune_lite.pth
│       ├── checkpoint_finetune.pth
│       └── checkpoint_resume.pth
├── test_data/
│   └── test_images/              # Ảnh cần chạy dự đoán bằng basnet_test.py
├── train_data/                   # Dữ liệu huấn luyện
├── validation_data/              # Dữ liệu validation/evaluation
├── BASNet_training.ipynb
├── basnet_bce_only.ipynb
├── basnet_ssim_only.ipynb
├── basnet_train.py               # Script train/fine-tune
├── basnet_evaluate.py            # Script đánh giá model
├── basnet_test.py                # Script dự đoán ảnh test
├── data_loader.py                # Đọc dữ liệu, tiền xử lý ảnh/mask
└── README.md
```

Nếu sau khi clone repository mà thiếu các thư mục `saved_models/`, `train_data/`, `validation_data/` hoặc `test_data/`, hãy tạo thủ công:

```bash
mkdir -p saved_models/basnet_bsi
mkdir -p train_data
mkdir -p validation_data
mkdir -p test_data/test_images
```

Trên Windows PowerShell có thể tạo bằng:

```powershell
mkdir saved_models\basnet_bsi
mkdir train_data
mkdir validation_data
mkdir test_data\test_images
```

---

## 2. Tải và lưu model/checkpoint

Checkpoint sau fine-tune được lưu tại Google Drive:

[Link chứa model sau fine-tune](https://drive.google.com/drive/folders/1fxyhRmLCfPovoN-WosHzqXmPUYIw66bS?usp=drive_link)

Trong Google Drive có thể thấy các file/folder như:

```text
code fine tune/
Kết quả fine tune/
model trước fine tune/
basnet_best_mae.pth
basnet_best_sm.pth
basnet_best_valloss.pth
basnet_best_wfm.pth
basnet_epoch_best.pth
```

Cách lưu đúng:

```text
Các file .pth sau fine-tune     -> saved_models/basnet_bsi/
Folder model trước fine tune    -> dùng khi muốn chạy lại từ checkpoint cũ hoặc pretrain
Folder Kết quả fine tune        -> chỉ dùng để tham khảo kết quả, không bắt buộc đưa vào project
Folder code fine tune           -> chỉ dùng nếu muốn đối chiếu hoặc thay thế code fine-tune
```

Ví dụ sau khi tải model, thư mục local cần có dạng:

```text
saved_models/
└── basnet_bsi/
    ├── basnet_best_mae.pth
    ├── basnet_best_sm.pth
    ├── basnet_best_valloss.pth
    ├── basnet_best_wfm.pth
    └── basnet_epoch_best.pth
```

Lưu ý quan trọng:

- Không đặt file `.pth` vào thư mục `model/` vì `model/` chỉ chứa code kiến trúc mạng.
- Nếu chỉ muốn chạy dự đoán hoặc đánh giá, nên dùng `basnet_best_mae.pth` hoặc `basnet_best_wfm.pth`.
- Nếu chạy code báo lỗi không tìm thấy checkpoint, kiểm tra lại đường dẫn trong `basnet_evaluate.py` hoặc `basnet_test.py` có trỏ đến `saved_models/basnet_bsi/*.pth` chưa.

---

## 3. Tải và lưu dataset

Project sử dụng dữ liệu cho bài toán **Salient Object Detection**. Có thể tải dataset từ các nguồn sau:

### 3.1. DUTS dataset

DUTS gồm tập train **DUTS-TR** và tập test/evaluation **DUTS-TE**.

Trang tải chính thức:

- [DUTS official website](https://saliencydetection.net/duts/)

Link tải trực tiếp:

- [DUTS-TR.zip](https://saliencydetection.net/duts/download/DUTS-TR.zip)
- [DUTS-TE.zip](https://saliencydetection.net/duts/download/DUTS-TE.zip)

Nếu link chính thức tải chậm, có thể dùng bản mirror trên Kaggle:

- [DUTS Saliency Detection Dataset - Kaggle](https://www.kaggle.com/datasets/balraj98/duts-saliency-detection-dataset)

Sau khi tải và giải nén, đặt dữ liệu theo cấu trúc:

```text
train_data/
└── DUTS-TR/
    ├── DUTS-TR-Image/            # Ảnh train .jpg
    └── DUTS-TR-Mask/             # Mask train .png

validation_data/
└── DUTS-TE/
    ├── DUTS-TE-Image/            # Ảnh validation/test .jpg
    └── DUTS-TE-Mask/             # Mask validation/test .png
```

Nếu code của bạn đang khai báo đường dẫn trực tiếp đến `train_data/DUTS-TR-Image/` và `train_data/DUTS-TR-Mask/`, hãy chuyển hai thư mục đó ra ngoài một cấp hoặc sửa lại đường dẫn trong `basnet_train.py` cho khớp với cấu trúc trên.

---

## 4. Chuẩn bị dữ liệu test riêng

Nếu chỉ muốn chạy thử model với ảnh bất kỳ, đặt ảnh cần dự đoán vào:

```text
test_data/
└── test_images/
    ├── image_01.jpg
    ├── image_02.png
    └── ...
```

Kết quả dự đoán sẽ được lưu theo đường dẫn khai báo trong `basnet_test.py`. Nếu code đang dùng thư mục `test_data/test_results/`, hãy tạo thư mục này trước khi chạy:

```bash
mkdir -p test_data/test_results
```

Trên Windows PowerShell:

```powershell
mkdir test_data\test_results
```

---

## 5. Cài đặt môi trường

Khuyến nghị dùng Python 3.8 trở lên. Tạo môi trường ảo:

```bash
python -m venv venv
```

Kích hoạt môi trường ảo:

```bash
# Windows
venv\Scripts\activate

# Linux/MacOS
source venv/bin/activate
```

Cài đặt thư viện cần thiết:

```bash
pip install torch torchvision numpy pillow opencv-python scikit-image scipy matplotlib tqdm
```

Nếu chạy bằng notebook, cài thêm:

```bash
pip install notebook ipykernel
```

---

## 6. Hướng dẫn chạy code

### 6.1. Train hoặc fine-tune model

Trước khi train, cần bảo đảm đã có dữ liệu trong:

```text
train_data/DUTS-TR/
validation_data/DUTS-TE/
```

Sau đó chạy:

```bash
python basnet_train.py
```

Checkpoint sau khi train/fine-tune nên được lưu vào:

```text
saved_models/basnet_bsi/
```

Nếu muốn fine-tune từ checkpoint có sẵn, hãy kiểm tra biến đường dẫn checkpoint trong `basnet_train.py`, ví dụ nên trỏ đến:

```text
saved_models/basnet_bsi/basnet_best_mae.pth
```

hoặc checkpoint phù hợp khác.

---

### 6.2. Đánh giá model

Để đánh giá model trên tập có ground truth, chạy:

```bash
python basnet_evaluate.py
```

Trước khi chạy, cần có:

```text
validation_data/DUTS-TE/           # Hoặc validation_data/ECSSD/
saved_models/basnet_bsi/*.pth      # Checkpoint cần đánh giá
```

Các chỉ số đánh giá thường dùng trong project:

- MAE
- S-measure
- E-measure
- Weighted F-measure
- Boundary F-measure

Nếu muốn đánh giá checkpoint tốt nhất theo MAE, dùng:

```text
saved_models/basnet_bsi/basnet_best_mae.pth
```

Nếu muốn đánh giá checkpoint tốt nhất theo Weighted F-measure, dùng:

```text
saved_models/basnet_bsi/basnet_best_wfm.pth
```

---

### 6.3. Chạy dự đoán ảnh test

Đặt ảnh cần dự đoán vào:

```text
test_data/test_images/
```

Sau đó chạy:

```bash
python basnet_test.py
```

Nếu chưa có checkpoint, tải checkpoint từ Google Drive ở phần 2 và đặt vào:

```text
saved_models/basnet_bsi/
```

---

### 6.4. Chạy bằng Jupyter Notebook

Có thể mở notebook bằng lệnh:

```bash
jupyter notebook BASNet_training.ipynb
```

Một số notebook khác trong project:

```text
basnet_bce_only.ipynb      # Thử nghiệm BCE loss
basnet_ssim_only.ipynb     # Thử nghiệm SSIM loss
```

---

## 7. Lỗi thường gặp

### Lỗi thiếu checkpoint `.pth`

Nguyên nhân: chưa tải model từ Google Drive hoặc lưu sai thư mục.

Cách sửa:

```text
Tải file .pth từ Google Drive
-> đặt vào saved_models/basnet_bsi/
-> kiểm tra lại đường dẫn checkpoint trong file .py đang chạy
```

---

### Lỗi thiếu dataset

Nguyên nhân: chưa tải DUTS/ECSSD hoặc đặt sai vị trí thư mục.

Cách sửa:

```text
DUTS-TR -> train_data/DUTS-TR/
DUTS-TE -> validation_data/DUTS-TE/
ECSSD   -> validation_data/ECSSD/
Ảnh test riêng -> test_data/test_images/
```

---

### Lỗi sai đường dẫn ảnh/mask

Nguyên nhân: tên thư mục sau khi giải nén khác với tên trong code.

Cách sửa:

- Kiểm tra lại `data_loader.py`.
- Kiểm tra lại đường dẫn trong `basnet_train.py`, `basnet_evaluate.py`, `basnet_test.py`.
- Đổi tên thư mục hoặc sửa biến path trong code cho khớp.

---

## 8. Tóm tắt nhanh cho người mới chạy

Thực hiện theo thứ tự:

```text
1. Clone repository từ GitHub.
2. Tạo các thư mục còn thiếu: train_data, validation_data, test_data/test_images, saved_models/basnet_bsi.
3. Tải model .pth từ Google Drive.
4. Đặt model .pth vào saved_models/basnet_bsi/.
5. Tải DUTS-TR và DUTS-TE.
6. Đặt DUTS-TR vào train_data/DUTS-TR/.
7. Đặt DUTS-TE vào validation_data/DUTS-TE/.
8. Nếu cần đánh giá ECSSD, tải ECSSD và đặt vào validation_data/ECSSD/.
9. Cài thư viện Python.
10. Chạy python basnet_train.py, python basnet_evaluate.py hoặc python basnet_test.py tùy mục đích.
```

