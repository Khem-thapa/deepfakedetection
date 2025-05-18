import os
import random
import shutil

def balance_fake_images(source_dir, target_dir, num_samples=759, seed=42):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    all_images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(seed)
    selected_images = random.sample(all_images, min(num_samples, len(all_images)))

    for img in selected_images:
        src_path = os.path.join(source_dir, img)
        dst_path = os.path.join(target_dir, img)
        shutil.copy(src_path, dst_path)

    print(f"[INFO] Copied {len(selected_images)} images to {target_dir}")


# Example usage
source_fake_dir = 'dataset/DFDC/fake/'     # Folder with all 3000 fake images
balanced_fake_dir = 'dataset/DFDC/fake_balanced/'  # Folder to copy 800 images to

balance_fake_images(source_fake_dir, balanced_fake_dir, num_samples=759)
