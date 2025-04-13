import os
import random
import matplotlib.pyplot as plt
import cv2

def show_image_comparison(base_path, img_name):
    """Display corresponding images from different lighting conditions."""
    plt.figure(figsize=(12, 4))
    categories = ["Low", "Normal", "Bright"]
    
    for idx, category in enumerate(categories):
        img_path = os.path.join(base_path, category, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, 3, idx + 1)
        plt.imshow(img)
        plt.title(f"{category} Light")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

base_path = "dataset/LOL-v2"
datasets = ["Real_captured", "Synthetic"]

for dataset in datasets:
    print(f"\nComparing {dataset} images:")
    path = os.path.join(base_path, dataset, "Train")
    
    if not os.path.exists(path):
        print(f"Warning: Dataset path not found: {path}")
        continue
    
    normal_dir = os.path.join(path, "Normal")
    if not os.path.exists(normal_dir):
        print(f"Warning: Normal directory not found: {normal_dir}")
        continue
        
    if dataset == "Real_captured":
        image_files = [f for f in os.listdir(normal_dir) if f.startswith('image')]
    else:  # Synthetic dataset
        image_files = [f for f in os.listdir(normal_dir) if f.startswith('r')]
        
    if not image_files:
        print(f"Warning: No images found in {normal_dir}")
        continue
        
    sample_files = random.sample(image_files, min(3, len(image_files)))
    
    for img_name in sample_files:
        print(f"Comparing {img_name}")
        show_image_comparison(path, img_name)
