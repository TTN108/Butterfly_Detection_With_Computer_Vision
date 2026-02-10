# Butterfly Image Classification (Kaggle)

Mẫu code này giúp bạn train và dự đoán cho bộ dữ liệu:
`phucthaiv02/butterfly-image-classification`

## 1) Chuẩn bị dữ liệu

Sau khi add dataset vào Kaggle Notebook, bạn có thể dùng:

```python
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

Mục tiêu là tìm được thư mục có cấu trúc `ImageFolder` (ví dụ):

```text
/kaggle/input/<ten-dataset>/train/
  class_1/
    img1.jpg
    ...
  class_2/
    ...
```

> Script `train_butterfly.py` đã hỗ trợ `--data_dir auto` để tự dò đường dẫn phổ biến trong Kaggle.

## 2) Cài thư viện

```bash
pip install -r requirements.txt
```

## 3) Train model

### Cách 1: auto-detect trong Kaggle
```bash
python train_butterfly.py --data_dir auto --epochs 10 --batch_size 32 --output_dir /kaggle/working/checkpoints
```

### Cách 2: chỉ định thủ công đường dẫn vừa tìm được
```bash
python train_butterfly.py --data_dir /kaggle/input/<ten-dataset>/train --epochs 10 --batch_size 32 --output_dir /kaggle/working/checkpoints
```

Các file sẽ được lưu trong `output_dir`:
- `resnet18_butterfly.pth`
- `labels.json`

## 4) Dự đoán ảnh mới

```bash
python predict.py \
  --image /kaggle/input/<ten-dataset>/test/some_image.jpg \
  --checkpoint /kaggle/working/checkpoints/resnet18_butterfly.pth \
  --labels /kaggle/working/checkpoints/labels.json
```

Ví dụ fine-tune toàn bộ backbone:

```bash
python train_butterfly.py --data_dir auto --epochs 12 --unfreeze --lr 1e-4
```
