import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatching(nn.Module):
    """Flow matching for DAC embeddings (B, T, latent_dim)"""

    def __init__(self, dit_model):
        super().__init__()
        self.dit = dit_model

    
    def compute_loss(self, x0, x1, genre_ids, mask=None): 
        """
        Compute flow matching training loss with optional masking for variable lengths.
        
        Args:
            x0: (B, T, latent_dim) source DAC embeddings
            x1: (B, T, latent_dim) target DAC embeddings
            genre_ids: (B,) target genre indices (0=classical, 1=rock, 2=unknown)
            mask: (B, T) optional mask (1.0 for valid, 0.0 for padding)
        
        Returns:
            loss: scalar MSE loss between predicted and true velocity
        """
        batch_size = x0.size(0)
        device = x0.device
        
        # Sample random timesteps
        t = torch.rand(batch_size, device=device)
        
        # Linear interpolation path: x(t) = (1-t)*x0 + t*x1
        # Reshape for broadcasting to (B, T, D) - 3D tensors
        t_expanded = t.view(batch_size, 1, 1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        
        # True velocity (constant for linear path)
        v_true = x1 - x0
        
        # Predicted velocity from DiT
        v_pred = self.dit(xt, t, genre_ids)
        
        # Compute loss (element-wise MSE)
        loss_per_element = F.mse_loss(v_pred, v_true, reduction='none')  # (B, T, D)
        
        if mask is not None:
            # Apply mask: (B, T) -> (B, T, 1) for broadcasting
            mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
            num_valid = mask_expanded.sum()
            if num_valid > 0:
                masked_loss = (loss_per_element * mask_expanded).sum() / num_valid
                return masked_loss
            return torch.tensor(0.0, device=loss_per_element.device)
        else:
            # No mask: standard MSE
            return loss_per_element.mean()
    
    def sample_euler(self, x0, genre_ids, num_steps=50):
        """
        Sample from the flow model using Euler integration.
        Transforms source DAC embeddings to target genre.
        
        Args:
            x0: (B, T, latent_dim) source DAC embeddings
            genre_ids: int or (B,) target genre (0=classical, 1=rock, 2=unknown)
            num_steps: number of integration steps
        
        Returns:
            (B, T, latent_dim) transformed DAC embeddings
        """
        batch_size = x0.size(0)
        device = x0.device
        
        if not isinstance(genre_ids, torch.Tensor):
            if isinstance(genre_ids, int):
                genre_ids = torch.tensor([genre_ids] * batch_size, device=device, dtype=torch.long)
            else:
                genre_ids = torch.tensor(genre_ids, device=device, dtype=torch.long)
        
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
            x0: (B, T, latent_dim) source DAC embeddings
            genre_ids: int or (B,) target genre (0=classical, 1=rock, 2=unknown)
            num_steps: number of integration steps
        
        Returns:
            (B, T, latent_dim) transformed DAC embeddings
        """
        batch_size = x0.size(0)
        device = x0.device

        if not isinstance(genre_ids, torch.Tensor):
            if isinstance(genre_ids, int):
                genre_ids = torch.tensor([genre_ids] * batch_size, device=device, dtype=torch.long)
            else:
                genre_ids = torch.tensor(genre_ids, device=device, dtype=torch.long)
        
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

    def sample(self, x0, genre_ids, mask=None, num_steps=50, method="euler"):
        """Convenience wrapper for sampling."""
        if method == "euler":
            return self.sample_euler(x0, genre_ids, num_steps=num_steps)
        if method == "heun":
            return self.sample_heun(x0, genre_ids, num_steps=num_steps)
        raise ValueError(f"Unknown sampling method: {method}")
    
    def forward(self, x0, x1, genre_ids):
        return self.compute_loss(x0, x1, genre_ids)
    

