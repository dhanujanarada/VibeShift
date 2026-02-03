import torch
import torch.nn as nn
import torch.nn.functional as F
from models.genre_classifier_loss import GenreClassifierAuxiliaryLoss
from typing import Optional, Tuple, Dict

class FlowMatching(nn.Module):

    def __init__(
        self,
        dit_model,
        use_genre_loss: bool = True,
        genre_loss_weight: float = 0.5,
        num_genres: int = 2,
        mel_height: int = 100,
        mel_width: int = 512,
    ):
        super().__init__()
        self.dit = dit_model
        self.use_genre_loss = use_genre_loss
        self.genre_loss_weight = genre_loss_weight
        
        # Initialize frozen genre classifier loss
        if self.use_genre_loss:
            self.genre_classifier_loss = GenreClassifierAuxiliaryLoss(
                num_genres=num_genres,
                mel_height=mel_height,
                mel_width=mel_width,
                embedding_dim=128,
                loss_weight=genre_loss_weight,
            )
        else:
            self.genre_classifier_loss = None

    
    def compute_loss(self, x0, x1, genre_ids): 
        """
        Compute flow matching training loss with optional genre classifier auxiliary loss.
        
        Args:
            x0: (B, C, n_mels, time_steps) source mel spectrograms
            x1: (B, C, n_mels, time_steps) target rock mel spectrograms
            genre_ids: (B,) target genre indices (1 for rock)
        
        Returns:
            loss: scalar MSE loss between predicted and true velocity
            loss_dict: Dictionary with loss components (if genre_loss enabled)
        """
        batch_size = x0.size(0)
        device = x0.device
        
        # Sample random timesteps
        t = torch.rand(batch_size, device=device)
        
        # Linear interpolation path: x(t) = (1-t)*x0 + t*x1
        t_expanded = t.view(batch_size, 1, 1, 1)  # Reshape for broadcasting to (B, C, H, W)
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        
        # True velocity (constant for linear path)
        v_true = x1 - x0
        
        # Predicted velocity from DiT
        v_pred = self.dit(xt, t, genre_ids)
        
        # Flow matching MSE loss
        flow_loss = F.mse_loss(v_pred, v_true)
        
        loss_dict = {'flow_matching_loss': flow_loss.item()}
        
        # Add genre classifier loss if enabled
        if self.use_genre_loss and self.genre_classifier_loss is not None:
            try:
                genre_loss, genre_loss_dict = self.genre_classifier_loss(xt, x1, genre_ids)
                flow_loss = flow_loss + genre_loss
                loss_dict.update(genre_loss_dict)
            except Exception as e:
                print(f"Warning: Genre classifier loss failed: {e}")
        
        return flow_loss, loss_dict
    
    def sample_euler(self, x0, genre_ids, num_steps=50):
        """
        Sample from the flow model using Euler integration.
        Transforms source mel spectrograms to target genre.
        
        Args:
            x0: (B, C, n_mels, time_steps) source mel spectrograms
            genre_ids: int or (B,) target genre (1 for rock)
            num_steps: number of integration steps
        
        Returns:
            (B, C, n_mels, time_steps) transformed mel spectrograms
        """
        batch_size = x0.size(0)
        device = x0.device
        
        if isinstance(genre_ids, int):
            genre_ids = torch.tensor([genre_ids] * batch_size, device=device)
        
        dt = 1.0 / num_steps
        xt = x0.clone()
        
        # Integrate the ODE: dx/dt = v(x, t)
        for step in range(num_steps):
            t = torch.full((batch_size,), step / num_steps, device=device)
            
            with torch.no_grad():
                v = self.dit(xt, t, genre_ids)  # Predict velocity field
            
            # Euler step
            xt = xt + v * dt
        
        return xt
    
    def sample_heun(self, x0, genre_ids, num_steps=50):
        """
        Sample using Heun's method (2nd order) for better accuracy.
        
        Args:
            x0: (B, C, n_mels, time_steps) source mel spectrograms
            genre_ids: int or (B,) target genre (1 for rock)
            num_steps: number of integration steps
        
        Returns:
            (B, C, n_mels, time_steps) transformed mel spectrograms
        """
        batch_size = x0.size(0)
        device = x0.device

        if isinstance(genre_ids, int):
            genre_ids = torch.tensor([genre_ids] * batch_size, device=device)
        
        dt = 1.0 / num_steps
        xt = x0.clone()

        for step in range(num_steps):
            t = torch.full((batch_size,), step / num_steps, device=device)
            
            with torch.no_grad():
                # First estimate
                v1 = self.dit(xt, t, genre_ids)
                xt_euler = xt + v1 * dt
                
                # Second estimate
                t_next = torch.full((batch_size,), (step + 1) / num_steps, device=device)
                v2 = self.dit(xt_euler, t_next, genre_ids)
            
            # Heun's method: average of two slopes
            xt = xt + (v1 + v2) * (dt / 2)
        
        return xt
    
    def forward(self, x0, x1, genre_ids):
        return self.compute_loss(x0, x1, genre_ids)
    
    def to(self, device):
        """Override to() to move all submodules including genre classifier loss."""
        self.dit = self.dit.to(device)
        if self.genre_classifier_loss is not None:
            self.genre_classifier_loss = self.genre_classifier_loss.to(device)
        return super().to(device)
    

