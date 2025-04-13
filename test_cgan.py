import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
from train_cgan import Generator

def process_image(generator, image_path, condition, output_path):
    """Process a single image with the trained generator."""
    # Load image and store original size
    input_img = Image.open(image_path).convert('RGB')
    original_size = input_img.size
    
    # Define transforms for model input (square 256x256)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Process image through model
    input_tensor = transform(input_img).unsqueeze(0)
    condition_tensor = torch.tensor([[condition]], dtype=torch.float32)
    
    # Move to same device as model
    device = next(generator.parameters()).device
    input_tensor = input_tensor.to(device)
    condition_tensor = condition_tensor.to(device)
    
    # Generate image
    with torch.no_grad():
        output_tensor = generator(input_tensor, condition_tensor)
    
    # Post-process
    output_tensor = output_tensor.cpu().squeeze(0)
    output_tensor = output_tensor * 0.5 + 0.5
    output_tensor = torch.clamp(output_tensor, 0, 1)
    
    # Convert to PIL and resize back to original dimensions
    output_img = transforms.ToPILImage()(output_tensor)
    output_img = output_img.resize(original_size, Image.Resampling.LANCZOS)
    output_img.save(output_path)
    print(f"Processed image saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Test Conditional GAN model')
    parser.add_argument('--input', type=str, required=True, help='Path to input normal-light image')
    parser.add_argument('--output', type=str, required=True, help='Path to save output image')
    parser.add_argument('--condition', type=float, default=1.0, 
                      help='Condition value: -1.0 for low light, 1.0 for bright')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                      help='Path to model checkpoint')
    args = parser.parse_args()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    
    checkpoint = torch.load(args.model, map_location=device)
    generator.load_state_dict(checkpoint['generator_state'])
    generator.eval()
    
    # Process image
    process_image(generator, args.input, args.condition, args.output)

if __name__ == "__main__":
    main()

# Example usage commands:
# Convert to bright image:
# python test_cgan.py --input sample/lion.jpeg --output sample/bright_lion.jpeg --condition 1.0
#
# Convert to low light image:
# python test_cgan.py --input sample/lion.jpeg --output sample/low_lion.jpeg --condition -1.0
