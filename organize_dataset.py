import os
import shutil

# Root dataset directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

CLASSES = ["Chickenpox", "Cowpox", "Healthy", "HFMD", "Measles", "Monkeypox"]

print("Starting automatic dataset reorganization...")

copied_files = set()
total_copied = 0

# FOLDS ফোল্ডারের পাথ
folds_dir = os.path.join(DATASET_DIR, "Datasets", "Dataset 02", "Original Images", "Original Images", "FOLDS")

if not os.path.exists(folds_dir):
    folds_dir = DATASET_DIR  # Fallback

for root, dirs, files in os.walk(folds_dir):
    # অগমেন্টেড বা ফিল্টার করা ফোল্ডার স্কিপ করা
    if "FOLDS_AUG" in root or "Augmented" in root:
        continue
    
    # বর্তমান ফোল্ডারটি ৬টি ক্লাসের কোনটি কিনা চেক করা
    folder_name = os.path.basename(root)
    if folder_name in CLASSES:
        dest_cls_dir = os.path.join(DATASET_DIR, folder_name)
        os.makedirs(dest_cls_dir, exist_ok=True)
        
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                # ডুপ্লিকেট ছবি এড়াতে ইউনিক ফাইলের নাম চেক
                if f not in copied_files:
                    src_file = os.path.join(root, f)
                    dest_file = os.path.join(dest_cls_dir, f)
                    shutil.copy2(src_file, dest_file)
                    copied_files.add(f)
                    total_copied += 1

print(f"\nDone! Successfully extracted and organized {total_copied} unique images.\n")
print("Class summary:")
for cls in CLASSES:
    cls_path = os.path.join(DATASET_DIR, cls)
    count = len(os.listdir(cls_path)) if os.path.exists(cls_path) else 0
    print(f" - {cls}: {count} images")