import argparse
import os
import urllib.request
from PIL import Image, ImageEnhance
import torch
import torchvision.transforms as transforms
import shutil
from train_cgan import Generator

def load_generator(checkpoint_path, device):
    """Load the trained generator model."""
    if not os.path.exists(checkpoint_path):
        print(f"Downloading model to {checkpoint_path}")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/nandhu-44/Light-Level-cGAN/main/models/light_level_cGAN-v0.pth",
            checkpoint_path
        )
    generator = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator_state = checkpoint.get('generator_state', checkpoint)
    generator.load_state_dict(generator_state)
    generator.eval()
    return generator

def process_image(generator, input_path, condition, output_path, device, blend_alpha=None, enhance=False):
    """Process a single image with the cGAN, blend, or enhance for intermediate conditions."""
    # Load image and store original size
    input_img = Image.open(input_path).convert('RGB')
    original_size = input_img.size

    # Handle normal condition (0.0)
    if condition == 0.0:
        shutil.copy(input_path, output_path)
        print(f"Copied original image to {output_path} (condition=0.0)")
        return

    # Define transforms for model input
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Prepare input
    input_tensor = transform(input_img).unsqueeze(0).to(device)
    # Map condition to -1.0 or 1.0 (model only supports these)
    model_condition = -1.0 if condition < 0.0 else 1.0
    condition_tensor = torch.tensor([model_condition], dtype=torch.float32).to(device)

    # Generate image
    with torch.no_grad():
        output_tensor = generator(input_tensor, condition_tensor)
    
    # Post-process
    output_tensor = output_tensor.cpu().squeeze(0) * 0.5 + 0.5
    output_tensor = torch.clamp(output_tensor, 0, 1)
    output_img = transforms.ToPILImage()(output_tensor)
    output_img = output_img.resize(original_size, Image.Resampling.LANCZOS)

    # Blend with original image to simulate intermediate conditions
    if blend_alpha is not None and 0.0 < blend_alpha < 1.0:
        output_img = Image.blend(input_img, output_img, blend_alpha)
        print(f"Blended image with alpha={blend_alpha:.2f}")

    # Enhance brightness for intermediate conditions
    if enhance and condition != -1.0 and condition != 1.0:
        brightness_factor = 1.0 + condition * 0.5  # -1.0 -> 0.5, 0.5 -> 1.25, 0.7 -> 1.35
        output_img = ImageEnhance.Brightness(output_img).enhance(brightness_factor)
        print(f"Enhanced brightness with factor={brightness_factor:.2f}")

    output_img.save(output_path)
    print(f"Saved {output_path} with requested condition={condition:.2f}, used model condition={model_condition:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Test Conditional GAN for single image')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to save output image')
    parser.add_argument('--condition', type=float, default=1.0, 
                        help='Condition value: -1.0 to 1.0 (-1.0=low, 0.0=normal, 1.0=bright)')
    parser.add_argument('--model', type=str, default='models/light_level_cGAN-v0.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--blend', action='store_true', 
                        help='Blend with original image for intermediate conditions')
    parser.add_argument('--alpha-scale', type=float, default=1.0,
                        help='Scale factor for blending alpha (higher = more cGAN output)')
    parser.add_argument('--enhance', action='store_true',
                        help='Apply brightness enhancement for intermediate conditions')
    args = parser.parse_args()

    # Validate condition
    if not -1.0 <= args.condition <= 1.0:
        raise ValueError("Condition must be in [-1.0, 1.0]")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = load_generator(args.model, device)

    # Determine blending alpha for intermediate conditions
    blend_alpha = None
    if args.blend and args.condition != 0.0 and args.condition != -1.0 and args.condition != 1.0:
        if args.condition > 0.0:
            # Bright: blend original with bright output (condition=1.0)
            base_alpha = (args.condition / 1.0) ** 2  # Quadratic for more cGAN
            blend_alpha = min(base_alpha * args.alpha_scale, 0.95)  # Cap at 95%
        else:
            # Low: blend original with low output (condition=-1.0)
            base_alpha = (abs(args.condition) / 1.0) ** 2
            blend_alpha = min(base_alpha * args.alpha_scale, 0.95)

    # Process image
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    process_image(generator, args.input, args.condition, args.output, device, blend_alpha, args.enhance)

if __name__ == '__main__':
    main()