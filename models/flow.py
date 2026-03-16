import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


class FlowMatching(nn.Module):
    """Flow matching for DAC embeddings (B, T, latent_dim)"""

    def __init__(self, dit_model, cfg_dropout_prob: float = 0.1, std_penalty_weight: float = 0.1):
        
        super().__init__()
        self.dit = dit_model
        self.cfg_dropout_prob = cfg_dropout_prob
        self.std_penalty_weight = std_penalty_weight
        
        if cfg_dropout_prob == 0.0:
            warnings.warn(
                "cfg_dropout_prob is 0.0. CFG training is disabled. "
                "The model will not learn unconditional generation, "
                "and CFG at inference will fail. "
                "Set cfg_dropout_prob >= 0.05 to enable CFG.",
                UserWarning,
            )

    
    def compute_loss(self, x0, x1, genre_ids, mask=None): 
        
        batch_size = x0.size(0)
        device = x0.device

        # CFG dropout: randomly replace genre conditioning with the null token.
        # Only applied during training so inference is unaffected.
        if self.cfg_dropout_prob > 0.0 and self.training:
            drop_mask = torch.rand(batch_size, device=device) < self.cfg_dropout_prob
            null_id = self.dit.null_genre_id
            genre_ids = torch.where(
                drop_mask,
                torch.full_like(genre_ids, null_id),
                genre_ids,
            )
        
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
                mse_loss = (loss_per_element * mask_expanded).sum() / num_valid
            else:
                return torch.tensor(0.0, device=loss_per_element.device)
        else:
            # No mask: standard MSE
            mse_loss = loss_per_element.mean()

        
        if self.std_penalty_weight > 0.0:
            std_penalty = F.mse_loss(v_pred.std(dim=1), v_true.std(dim=1))
            return mse_loss + self.std_penalty_weight * std_penalty

        return mse_loss
    
    def sample_euler(self, x0, genre_ids, num_steps=50, guidance_scale: float = 1.0):
        
        batch_size = x0.size(0)
        device = x0.device
        
        if not isinstance(genre_ids, torch.Tensor):
            if isinstance(genre_ids, int):
                genre_ids = torch.tensor([genre_ids] * batch_size, device=device, dtype=torch.long)
            else:
                genre_ids = torch.tensor(genre_ids, device=device, dtype=torch.long)
        
        dt = 1.0 / num_steps
        xt = x0.clone()
        
        # Pre-build null genre tensor once (reused every step)
        null_ids = torch.full_like(genre_ids, self.dit.null_genre_id) if guidance_scale != 1.0 else None

        # Integrate the ODE: dx/dt = v(x, t)
        for step in range(num_steps):
            t = torch.full((batch_size,), step / num_steps, device=device)
            
            with torch.no_grad():
                v_cond = self.dit(xt, t, genre_ids)  # conditioned velocity

                if guidance_scale != 1.0:
                    v_uncond = self.dit(xt, t, null_ids)  # unconditional velocity
                    # CFG: push further toward the target genre
                    v = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v = v_cond
            
            # Euler step
            xt = xt + v * dt
        
        return xt
    
    def sample_euler_cfg(self, x0, genre_ids, num_steps=50, guidance_scale=3.0):
        """
        Euler sampling with classifier-free guidance (CFG).
        Runs two forward passes per step: one conditioned, one unconditional.
        
        Args:
            x0: (B, T, latent_dim) source embeddings
            genre_ids: (B,) target genre indices
            num_steps: integration steps
            guidance_scale: CFG strength (>1.0 pushes toward target genre)
        
        Returns:
            (B, T, latent_dim) transformed embeddings
        """
        batch_size = x0.size(0)
        device = x0.device

        if not isinstance(genre_ids, torch.Tensor):
            if isinstance(genre_ids, int):
                genre_ids = torch.tensor([genre_ids] * batch_size, device=device, dtype=torch.long)
            else:
                genre_ids = torch.tensor(genre_ids, device=device, dtype=torch.long)

        # Null token is one past the last real genre
        null_ids = torch.full_like(genre_ids, self.dit.null_genre_id)

        dt = 1.0 / num_steps
        xt = x0.clone()

        for step in range(num_steps):
            t = torch.full((batch_size,), step / num_steps, device=device)

            with torch.no_grad():
                v_cond = self.dit(xt, t, genre_ids)
                v_uncond = self.dit(xt, t, null_ids)

            # CFG: interpolate between unconditional and conditioned velocity
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            xt = xt + v * dt

        return xt
    
    def sample_heun(self, x0, genre_ids, num_steps=50, guidance_scale: float = 1.0):
        """
        Sample using Heun's method (2nd order) for better accuracy.
        
        Args:
            x0: (B, T, latent_dim) source DAC embeddings
            genre_ids: int or (B,) target genre (0=classical, 1=rock, 2=unknown)
            num_steps: number of integration steps
            guidance_scale: CFG scale (1.0 = no guidance, >1 = stronger genre push)
        
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

        null_ids = torch.full_like(genre_ids, self.dit.null_genre_id) if guidance_scale != 1.0 else None

        for step in range(num_steps):
            t = torch.full((batch_size,), step / num_steps, device=device)
            
            with torch.no_grad():
                # First estimate
                v1_cond = self.dit(xt, t, genre_ids)
                if guidance_scale != 1.0:
                    v1_uncond = self.dit(xt, t, null_ids)
                    v1 = v1_uncond + guidance_scale * (v1_cond - v1_uncond)
                else:
                    v1 = v1_cond

                xt_euler = xt + v1 * dt

                # Second estimate
                t_next = torch.full((batch_size,), (step + 1) / num_steps, device=device)
                v2_cond = self.dit(xt_euler, t_next, genre_ids)
                if guidance_scale != 1.0:
                    v2_uncond = self.dit(xt_euler, t_next, null_ids)
                    v2 = v2_uncond + guidance_scale * (v2_cond - v2_uncond)
                else:
                    v2 = v2_cond
            
            # Heun's method: average of two slopes
            xt = xt + (v1 + v2) * (dt / 2)
        
        return xt
    
    def sample_heun_cfg(self, x0, genre_ids, num_steps=50, guidance_scale=3.0):
        """
        Heun's method sampling with classifier-free guidance (CFG).
        2nd-order integration with CFG double-forward-pass per step.
        
        Args:
            x0: (B, T, latent_dim) source embeddings
            genre_ids: (B,) target genre indices
            num_steps: integration steps
            guidance_scale: CFG strength (>1.0 pushes toward target genre)
        
        Returns:
            (B, T, latent_dim) transformed embeddings
        """
        batch_size = x0.size(0)
        device = x0.device

        if not isinstance(genre_ids, torch.Tensor):
            if isinstance(genre_ids, int):
                genre_ids = torch.tensor([genre_ids] * batch_size, device=device, dtype=torch.long)
            else:
                genre_ids = torch.tensor(genre_ids, device=device, dtype=torch.long)

        null_ids = torch.full_like(genre_ids, self.dit.null_genre_id)

        dt = 1.0 / num_steps
        xt = x0.clone()

        for step in range(num_steps):
            t = torch.full((batch_size,), step / num_steps, device=device)

            with torch.no_grad():
                # First estimate
                v1_cond = self.dit(xt, t, genre_ids)
                v1_uncond = self.dit(xt, t, null_ids)
                v1 = v1_uncond + guidance_scale * (v1_cond - v1_uncond)
                xt_euler = xt + v1 * dt

                # Second estimate
                t_next = torch.full((batch_size,), (step + 1) / num_steps, device=device)
                v2_cond = self.dit(xt_euler, t_next, genre_ids)
                v2_uncond = self.dit(xt_euler, t_next, null_ids)
                v2 = v2_uncond + guidance_scale * (v2_cond - v2_uncond)

            # Heun's method: average of two slopes
            xt = xt + (v1 + v2) * (dt / 2)

        return xt

    def sample(self, x0, genre_ids, mask=None, num_steps=50, method="euler", guidance_scale=1.0):
        """Convenience wrapper for sampling with optional CFG."""
        if guidance_scale > 1.0:
            if method == "euler":
                return self.sample_euler_cfg(x0, genre_ids, num_steps, guidance_scale)
            elif method == "heun":
                return self.sample_heun_cfg(x0, genre_ids, num_steps, guidance_scale)
        # Fallback to standard sampling without CFG
        if method == "euler":
            return self.sample_euler(x0, genre_ids, num_steps)
        elif method == "heun":
            return self.sample_heun(x0, genre_ids, num_steps)
        raise ValueError(f"Unknown sampling method: {method}")
    
    def forward(self, x0, x1, genre_ids, mask=None):
        return self.compute_loss(x0, x1, genre_ids, mask)
    

