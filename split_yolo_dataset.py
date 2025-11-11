import os
import random
import shutil
from pathlib import Path

dataset_dir = Path("project-tubers-dataset")
images_dir = dataset_dir / "images"
labels_dir = dataset_dir / "labels"

output_dir = Path("datasets/tubers")
train_img = output_dir / "images/train"
val_img = output_dir / "images/val"
train_lbl = output_dir / "labels/train"
val_lbl = output_dir / "labels/val"

for d in [train_img, val_img, train_lbl, val_lbl]:
    d.mkdir(parents=True, exist_ok=True)

images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
val_ratio = 0.2
val_count = int(len(images) * val_ratio)
val_samples = set(random.sample(images, val_count))

for img_path in images:
    label_path = labels_dir / f"{img_path.stem}.txt"
    if img_path in val_samples:
        shutil.copy(img_path, val_img / img_path.name)
        if label_path.exists():
            shutil.copy(label_path, val_lbl / label_path.name)
    else:
        shutil.copy(img_path, train_img / img_path.name)
        if label_path.exists():
            shutil.copy(label_path, train_lbl / label_path.name)

for file_name in ["classes.txt", "notes.json"]:
    src_file = dataset_dir / file_name
    if src_file.exists():
        shutil.copy(src_file, output_dir / file_name)

print("✅ Dataset dividido correctamente:")
print(f"Train: {len(images) - val_count} imágenes")
print(f"Val:   {val_count} imágenes")
