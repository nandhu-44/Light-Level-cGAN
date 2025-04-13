import os
import re

def rename_images(base_dir):
    """Rename all images in Low/Normal/Bright dirs to standard format."""
    categories = ["Low", "Normal", "Bright"]
    
    for category in categories:
        dir_path = os.path.join(base_dir, category)
        if not os.path.exists(dir_path):
            continue
            
        for old_name in os.listdir(dir_path):
            if not old_name.lower().endswith(('.png', '.jpg')):
                continue
            
            new_name = re.sub(r'^(low|normal|bright)', 'image', old_name.lower())
            
            old_path = os.path.join(dir_path, old_name)
            new_path = os.path.join(dir_path, new_name)
            
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_name} -> {new_name}")

if __name__ == "__main__":
    base_path = "dataset/LOL-v2"
    datasets = ["Real_captured"]
    splits = ["Train", "Test"]
    
    for dataset in datasets:
        for split in splits:
            path = os.path.join(base_path, dataset, split)
            if os.path.exists(path):
                print(f"\nProcessing {dataset}/{split}...")
                rename_images(path)
                
    print("\nRenaming completed!")
