# Butterfly Image Classification (Kaggle)

Mẫu code này giúp bạn train và dự đoán cho bộ dữ liệu:
`phucthaiv02/butterfly-image-classification`

## 1) Chuẩn bị dữ liệu

Sau khi tải dữ liệu từ Kaggle, đặt dữ liệu theo cấu trúc `ImageFolder`:

```text
dataset/
  train/
    class_1/
      img1.jpg
      ...
    class_2/
      ...
```

> `train_butterfly.py` mặc định đọc từ `dataset/train`.

## 2) Cài thư viện

```bash
pip install -r requirements.txt
```

## 3) Train model

```bash
python train_butterfly.py --data_dir dataset/train --epochs 10 --batch_size 32
```

Các file sẽ được lưu trong `checkpoints/`:
- `resnet18_butterfly.pth`
- `labels.json`

## 4) Dự đoán ảnh mới

```bash
python predict.py --image path/to/new_image.jpg
```

Ví dụ fine-tune toàn bộ backbone:

```bash
python train_butterfly.py --data_dir dataset/train --epochs 12 --unfreeze --lr 1e-4
```
