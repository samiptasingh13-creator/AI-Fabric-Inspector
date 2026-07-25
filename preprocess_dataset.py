import os
import cv2
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def enhance_fabric_image(image_path):
    """Applies CLAHE on L-channel + Non-Local Means Denoising."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # BGR -> LAB space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Recombine & Denoise
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    denoised = cv2.fastNlMeansDenoisingColored(
        enhanced_bgr, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21
    )
    return denoised

def process_single_item(item):
    img_path, img_dst_dir, lbl_dst_dir, label_files = item
    enhanced = enhance_fabric_image(img_path)
    if enhanced is not None:
        cv2.imwrite(str(img_dst_dir / img_path.name), enhanced)

    if img_path.stem in label_files:
        lbl_src = label_files[img_path.stem]
        shutil.copy(lbl_src, lbl_dst_dir / lbl_src.name)
    return True

def process_dataset():
    possible_roots = [
        Path("dataset_v2/dataset"),
        Path("dataset_v2"),
    ]
    
    src_root = None
    for p in possible_roots:
        if (p / "train").exists() or (p / "data.yaml").exists():
            src_root = p
            break
            
    if src_root is None:
        print("❌ Could not locate dataset root folder!")
        return

    print(f"\n[i] Dataset Root: '{src_root.resolve()}'")
    dst_root = Path("dataset_v2_clahe")
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    splits = [
        ('train', src_root / 'train'),
        ('val', src_root / 'valid' if (src_root / 'valid').exists() else src_root / 'val')
    ]

    # Use CPU threads available
    num_workers = os.cpu_count() or 4
    print(f"[+] Utilizing {num_workers} CPU threads for parallel processing...")

    for dst_split_name, split_dir in splits:
        if not split_dir.exists():
            continue

        img_dst_dir = dst_root / dst_split_name / 'images'
        lbl_dst_dir = dst_root / dst_split_name / 'labels'
        img_dst_dir.mkdir(parents=True, exist_ok=True)
        lbl_dst_dir.mkdir(parents=True, exist_ok=True)

        image_files = [p for p in split_dir.rglob('*') if p.is_file() and p.suffix.lower() in valid_exts]
        total_imgs = len(image_files)
        label_files = {p.stem: p for p in split_dir.rglob('*.txt')}

        print(f"\n[*] Processing '{dst_split_name}' set ({total_imgs} images)...")

        tasks = [(img, img_dst_dir, lbl_dst_dir, label_files) for img in image_files]

        completed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_item, task) for task in tasks]
            for future in as_completed(futures):
                completed += 1
                if completed % 50 == 0 or completed == total_imgs:
                    print(f"    Progress: [{completed}/{total_imgs}] images completed...")

    # Copy data.yaml
    yaml_files = list(src_root.glob("*.yaml"))
    if yaml_files:
        shutil.copy(yaml_files[0], dst_root / 'data.yaml')
        print(f"\n[✔] Copied data.yaml to '{dst_root / 'data.yaml'}'")

    print(f"\n[✔] SUCCESS: Dataset preprocessing complete!")

if __name__ == "__main__":
    process_dataset()