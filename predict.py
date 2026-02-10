import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


def load_model(ckpt_path: str, num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(model.fc.in_features, num_classes),
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def predict(image_path: str, ckpt_path: str, labels_path: str, top_k: int = 3):
    with open(labels_path, "r", encoding="utf-8") as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}

    model = load_model(ckpt_path, len(idx_to_class))

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        vals, inds = torch.topk(probs, k=top_k, dim=1)

    results = []
    for p, idx in zip(vals[0].tolist(), inds[0].tolist()):
        results.append((idx_to_class[idx], p))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/resnet18_butterfly.pth")
    parser.add_argument("--labels", default="checkpoints/labels.json")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    outputs = predict(args.image, args.checkpoint, args.labels, args.top_k)
    for i, (name, prob) in enumerate(outputs, start=1):
        print(f"Top {i}: {name} ({prob:.4f})")


if __name__ == "__main__":
    main()
