import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

# Custom Dataset
class LOLv2Dataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.base_path = base_path
        self.transform = transform
        
        # Collect image paths from Real_captured and Synthetic
        self.normal_paths = []
        self.lowlight_paths = []
        self.bright_paths = []
        
        for data_type in ["Real_captured", "Synthetic"]:
            normal_dir = os.path.join(base_path, data_type, "Train/Normal")
            lowlight_dir = os.path.join(base_path, data_type, "Train/Low")
            bright_dir = os.path.join(base_path, data_type, "Train/Bright")
            
            normal_images = sorted(os.listdir(normal_dir))
            for img_name in normal_images:
                self.normal_paths.append(os.path.join(normal_dir, img_name))
                self.lowlight_paths.append(os.path.join(lowlight_dir, img_name))
                self.bright_paths.append(os.path.join(bright_dir, img_name))

    def __len__(self):
        return len(self.normal_paths)

    def __getitem__(self, idx):
        # Load images
        normal_img = Image.open(self.normal_paths[idx]).convert('RGB')
        lowlight_img = Image.open(self.lowlight_paths[idx]).convert('RGB')
        bright_img = Image.open(self.bright_paths[idx]).convert('RGB')

        # Apply transforms
        if self.transform:
            normal_img = self.transform(normal_img)
            lowlight_img = self.transform(lowlight_img)
            bright_img = self.transform(bright_img)

        # Randomly choose low-light or bright-light target
        condition = torch.rand(1).item() * 2 - 1  # -1 to 1
        target_img = lowlight_img if condition < 0 else bright_img
        condition = torch.tensor([condition], dtype=torch.float32)

        return normal_img, target_img, condition

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(4, 64, 4, stride=2, padding=1)  # 3 channels + 1 condition
        self.enc2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        # Decoder
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x, condition):
        cond = condition.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        x = torch.cat([x, cond], dim=1)
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        d1 = self.relu(self.dec1(e3))
        d2 = self.relu(self.dec2(d1))
        out = self.tanh(self.dec3(d2))
        return out

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, 4, stride=2, padding=1)  # 3 channels + 1 condition
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 1, 4, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, condition):
        cond = condition.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        x = torch.cat([x, cond], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        out = self.conv4(x)
        return out

# Training function
def train_cgan(base_path, num_epochs=100, batch_size=16, resume=False):
    # Create checkpoint directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Transformations and dataset setup
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = LOLv2Dataset(base_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # Initialize tracking variables
    start_epoch = 0
    best_g_loss = float('inf')
    
    # Resume from checkpoint if requested
    if resume:
        checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            generator.load_state_dict(checkpoint['generator_state'])
            discriminator.load_state_dict(checkpoint['discriminator_state'])
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state'])
            start_epoch = checkpoint['epoch']
            best_g_loss = checkpoint.get('best_g_loss', float('inf'))
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("No checkpoint found, starting from scratch")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        g_losses = []
        d_losses = []
        
        for i, (normal_img, target_img, condition) in enumerate(dataloader):
            normal_img, target_img, condition = normal_img.to(device), target_img.to(device), condition.to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            real_output = discriminator(target_img, condition)
            fake_img = generator(normal_img, condition)
            fake_output = discriminator(fake_img.detach(), condition)
            d_loss = criterion(real_output, torch.ones_like(real_output)) + criterion(fake_output, torch.zeros_like(fake_output))
            d_loss.backward()
            d_optimizer.step()
            d_losses.append(d_loss.item())

            # Train Generator
            g_optimizer.zero_grad()
            fake_output = discriminator(fake_img, condition)
            g_loss = criterion(fake_output, torch.ones_like(fake_output)) + 100 * l1_loss(fake_img, target_img)
            g_loss.backward()
            g_optimizer.step()
            g_losses.append(g_loss.item())

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] "
                      f"D_Loss: {d_loss.item():.4f} G_Loss: {g_loss.item():.4f}")

        # Calculate average losses for the epoch
        avg_g_loss = sum(g_losses) / len(g_losses)
        avg_d_loss = sum(d_losses) / len(d_losses)
        
        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'generator_state': generator.state_dict(),
            'discriminator_state': discriminator.state_dict(),
            'g_optimizer_state': g_optimizer.state_dict(),
            'd_optimizer_state': d_optimizer.state_dict(),
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
            'best_g_loss': best_g_loss
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, "latest_checkpoint.pth"))
        
        # Save best model (based on generator loss)
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"Saved best model with G_Loss: {avg_g_loss:.4f}")

if __name__ == "__main__":
    base_path = "dataset/LOL-v2/"
    train_cgan(base_path, resume=True)  # Set resume=True to continue from checkpoint