from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os


def create_bright_image(input_path, output_path):
    """Convert a normal-light image to a bright-light image."""
    img = Image.open(input_path).convert("RGB")

    enhancer = ImageEnhance.Brightness(img)
    bright_img = enhancer.enhance(1.6)  # Increased from 1.4

    enhancer = ImageEnhance.Contrast(bright_img)
    bright_img = enhancer.enhance(1.3)  # Increased from 1.2

    bright_np = np.array(bright_img)
    gamma = 0.7  # Decreased from 0.8
    bright_np = np.clip(((bright_np / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(bright_np, cv2.COLOR_RGB2BGR))


def generate_bright_dataset(normal_dir, bright_dir):
    """Process all normal images in a directory to create bright versions."""
    for img_name in os.listdir(normal_dir):
        input_path = os.path.join(normal_dir, img_name)
        output_path = os.path.join(bright_dir, img_name)
        create_bright_image(input_path, output_path)
        print(f"Created bright image: {output_path}")


base_path = "dataset/LOL-v2/"
dirs = [
    ("Real_captured/Train/Normal", "Real_captured/Train/Bright"),
    ("Synthetic/Train/Normal", "Synthetic/Train/Bright"),
    ("Real_captured/Test/Normal", "Real_captured/Test/Bright"),
    ("Synthetic/Test/Normal", "Synthetic/Test/Bright"),
]

for normal_dir, bright_dir in dirs:
    generate_bright_dataset(
        os.path.join(base_path, normal_dir), os.path.join(base_path, bright_dir)
    )

print("Bright-light images generated successfully.")
