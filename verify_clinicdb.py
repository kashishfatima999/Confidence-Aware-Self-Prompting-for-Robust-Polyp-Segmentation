
from pathlib import Path
from PIL import Image

root = Path("data/raw/cvc-clinicdb/PNG")
images_dir = root / "Original"
masks_dir = root / "Ground Truth"

images = {p.stem: p for p in images_dir.glob("*.png")}
masks = {p.stem: p for p in masks_dir.glob("*.png")}

print("Images:", len(images))
print("Masks:", len(masks))
print("Images without masks:", sorted(images.keys() - masks.keys()))
print("Masks without images:", sorted(masks.keys() - images.keys()))

bad_images = []
bad_masks = []
image_sizes = set()
mask_sizes = set()

for path in images.values():
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            image_sizes.add(im.size)
    except Exception:
        bad_images.append(path.name)

for path in masks.values():
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            mask_sizes.add(im.size)
    except Exception:
        bad_masks.append(path.name)

print("Image dimensions:", image_sizes)
print("Mask dimensions:", mask_sizes)
print("Unreadable images:", bad_images)
print("Unreadable masks:", bad_masks)