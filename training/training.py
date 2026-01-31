import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import sys
from pathlib import Path
from tqdm import tqdm
import wandb
from models.dit import DiT
from models.flow import FlowMatching

# filepath: c:\Users\Dhanuja\Desktop\Vibeshift\VibeShift\training\training.py
import torch.nn as nn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))



class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        lr=1e-4,
        num_epochs=100,
        checkpoint_dir='checkpoints',
        use_wandb=False,
        project_name='vibeshift'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        
        # Optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize wandb
        if use_wandb:
            wandb.init(project=project_name, config={
                'lr': lr,
                'num_epochs': num_epochs,
                'batch_size': train_loader.batch_size
            })
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch (adjust based on your dataset)
            x0 = batch['source'].to(self.device)  # Source mel spectrograms
            x1 = batch['target'].to(self.device)  # Target mel spectrograms
            genre_ids = batch['genre_id'].to(self.device)  # Target genre IDs
            
            # Forward pass
            loss = self.model(x0, x1, genre_ids)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'train_loss': loss.item(),
                    'epoch': epoch,
                    'step': epoch * len(self.train_loader) + batch_idx
                })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                x0 = batch['source'].to(self.device)
                x1 = batch['target'].to(self.device)
                genre_ids = batch['genre_id'].to(self.device)
                
                loss = self.model(x0, x1, genre_ids)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        if self.use_wandb:
            wandb.log({
                'val_loss': avg_loss,
                'epoch': epoch
            })
        
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f'Saved best model with val_loss: {val_loss:.4f}')
    
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f'Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f}')
            
            # Validate
            val_loss = self.validate(epoch)
            print(f'Epoch {epoch+1}/{self.num_epochs} - Val Loss: {val_loss:.4f}')
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        if self.use_wandb:
            wandb.finish()
        
        print(f'Training complete. Best val_loss: {best_val_loss:.4f}')


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize DiT model
    dit = DiT(
        use_mel_patches=True,
        patch_height=8,
        patch_width=8,
        in_channels=1
    )
    
    # Wrap with Flow Matching
    model = FlowMatching(dit)
    
    # TODO: Create your dataloaders
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=None,  # Replace with your train_loader
        val_loader=None,    # Replace with your val_loader
        device=device,
        lr=1e-4,
        num_epochs=100,
        checkpoint_dir='checkpoints',
        use_wandb=False
    )
    
    # Start training
    # trainer.train()


if __name__ == '__main__':
    main()