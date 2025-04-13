import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import argparse
from train_cgan import Generator

def generate_lighting_variations(generator, normal_img_path, output_low_path, output_bright_path, device):
    """Generate low-light and bright-light versions of a single image, preserving original dimensions."""
    # Load image and get original dimensions
    input_img = Image.open(normal_img_path).convert('RGB')
    original_size = input_img.size  # (width, height)
    
    # Define transformation for model input
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize for model
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Preprocess image
    input_tensor = transform(input_img).unsqueeze(0).to(device)
    
    # Generate images
    with torch.no_grad():
        # Low-light (condition = -1)
        condition = torch.tensor([-1.0], dtype=torch.float32).to(device)
        low_tensor = generator(input_tensor, condition)
        
        # Bright-light (condition = 1)
        condition = torch.tensor([1.0], dtype=torch.float32).to(device)
        bright_tensor = generator(input_tensor, condition)
    
    # Post-process outputs
    to_pil = transforms.ToPILImage()
    for tensor, output_path in [(low_tensor, output_low_path), (bright_tensor, output_bright_path)]:
        # Denormalize
        tensor = tensor.cpu().squeeze(0) * 0.5 + 0.5
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to PIL and resize to original dimensions
        output_img = to_pil(tensor)
        output_img = output_img.resize(original_size, Image.LANCZOS)  # High-quality resize
        output_img.save(output_path)
        print(f"Saved image: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Test Conditional GAN on a single image')
    parser.add_argument('--input', type=str, required=True, help='Path to input normal-light image')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save output images')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', 
                       help='Path to model checkpoint')
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    generator_state = checkpoint.get('generator_state', checkpoint)  # Handle both formats
    generator.load_state_dict(generator_state)
    generator.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output paths
    img_name = os.path.basename(args.input)
    output_low_path = os.path.join(args.output_dir, f"low_{img_name}")
    output_bright_path = os.path.join(args.output_dir, f"bright_{img_name}")
    
    # Process image
    generate_lighting_variations(generator, args.input, output_low_path, output_bright_path, device)

if __name__ == "__main__":
    main()