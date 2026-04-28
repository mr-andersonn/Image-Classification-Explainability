from pathlib import Path
import shutil

src_dir = Path(r"C:\Users\anton\Documents\DAT255\versions\2\Fish_Dataset\Fish_Dataset")

clean_dir = Path(r"C:\Users\anton\Documents\DAT255\versions\2\Fish_Dataset\Fish_Dataset_clean")

clean_dir.mkdir(parents=True, exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for class_folder in src_dir.iterdir():
    if not class_folder.is_dir():
        continue

    class_name = class_folder.name

    if "GT" in class_name:
        continue

    target_class_folder = clean_dir / class_name
    target_class_folder.mkdir(parents=True, exist_ok=True)

    for file in class_folder.iterdir():
        if file.is_file():
            if file.suffix.lower() in image_extensions and "GT" not in file.name:
                shutil.copy2(file, target_class_folder / file.name)

    nested_image_folder = class_folder / class_name

    if nested_image_folder.exists() and nested_image_folder.is_dir():
        for file in nested_image_folder.iterdir():
            if file.is_file():
                if file.suffix.lower() in image_extensions and "GT" not in file.name:
                    shutil.copy2(file, target_class_folder / file.name)

print("Clean dataset created successfully!")
print(f"Source folder: {src_dir}")
print(f"Clean folder:  {clean_dir}")

print("\nImages per class:")
for class_folder in clean_dir.iterdir():
    if class_folder.is_dir():
        count = len([
            file for file in class_folder.iterdir()
            if file.suffix.lower() in image_extensions
        ])
        print(f"{class_folder.name}: {count}")