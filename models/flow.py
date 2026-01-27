import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatching(nn.Module):

    def __init__(self, dit_model):
        super().__init__()
        self.dit = dit_model

    
    def compute_loss(self, x0, x1, genre_ids): 
        """
        Compute flow matching training loss.
        """
        batch_size = x0.size(0)
        device = x0.device
        
        t = torch.rand(batch_size, device=device)
        t_expanded = t.view(batch_size, 1, 1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        
        v_true = x1 - x0
        v_pred = self.dit(xt, t, genre_ids)
        
        loss = F.mse_loss(v_pred, v_true)
        return loss
    
    def sample_euler(self, x0, genre_ids, num_steps=50):
        """
        Sample from the flow model using Euler integration.
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
                
                v = self.dit(xt, t, genre_ids)  
            
           
            xt = xt + v * dt
        
        return xt
    
    def sample_heun(self, x0, genre_ids, num_steps=50):
        batch_size = x0.size(0)
        device = x0.device

        if isinstance(genre_ids, int):
            genre_ids = torch.tensor([genre_ids] * batch_size, device=device)
        
        dt = 1.0 / num_steps
        xt = x0.clone()

        for step in range(num_steps):
            t = torch.full((batch_size,), step / num_steps, device=device)
            
            with torch.no_grad():
                v1 = self.dit(xt, t, genre_ids)
                xt_euler = xt + v1 * dt
                t_next = torch.full((batch_size,), (step + 1) / num_steps, device=device)
                v2 = self.dit(xt_euler, t_next, genre_ids)
            
            xt = xt + (v1 + v2) * (dt / 2)
        return xt
    
    def forward(self, x0, x1, genre_ids):
        return self.compute_loss(x0, x1, genre_ids)
    

