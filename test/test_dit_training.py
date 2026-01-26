import torch
import torch.optim as optim
import sys
sys.path.append('c:\\Users\\Dhanuja\\Desktop\\Vibeshift\\VibeShift')

from models.dit import DiT

model = DiT(patch_dim=256, embed_dim=512, num_blocks=8, num_heads=8, num_genres=10)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("🚀 Starting training simulation...\n")

for epoch in range(5):
    total_loss = 0
    
    for step in range(10):
        # Simulate batch
        x = torch.randn(4, 100, 256)
        t = torch.rand(4)
        genres = torch.randint(0, 10, (4,))
        target = torch.randn(4, 100, 256)  # Target transformation
        
        # Forward
        output = model(x, t, genres)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"Epoch {epoch+1}/5 | Step {step+1}/10 | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / 10
    print(f"✅ Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}\n")

print("✅ Training simulation complete!")