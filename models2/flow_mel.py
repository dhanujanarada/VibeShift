import torch
import torch.nn as nn
import torch.nn.functional as F

from models2.dit_mel import DiTMel


class FlowMatchingMel(nn.Module):
    """
    Conditional Flow Matching for mel-spectrogram genre transfer.

    Training objective (OT straight-paths):
        x_t  = (1-t) * x0  +  t * x1
        v_t  = x1 - x0          ← target velocity (constant along the path)
        loss = MSE(model(x_t, t, genre), v_t)

    where:
        x0 — source mel  (e.g. non-rock, genre_id=0)
        x1 — target mel  (e.g. rock,     genre_id=1)
    """

    GENRE_MAP = {
        "punk":      1,
        "synth":     0,
    }

    def __init__(
        self,
        dit_model: DiTMel | None = None,
        std_penalty_weight: float = 0.1,
        # DiTMel constructor kwargs (used only when dit_model is None)
        n_mels: int = 100,
        patch_height: int = 10,
        patch_width: int = 16,
        embed_dim: int = 512,
        num_blocks: int = 8,
        num_heads: int = 8,
        num_genres: int = 2,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        if dit_model is not None:
            self.dit = dit_model
        else:
            self.dit = DiTMel(
                n_mels=n_mels,
                patch_height=patch_height,
                patch_width=patch_width,
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                num_heads=num_heads,
                num_genres=num_genres,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

        self.std_penalty_weight = std_penalty_weight

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_to_same(x0: torch.Tensor, x1: torch.Tensor):
        """Trim both mels to the same time length (the shorter of the two)."""
        T = min(x0.shape[-1], x1.shape[-1])
        return x0[..., :T], x1[..., :T]

    def _genre_tensor(
        self,
        genre,
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(genre, str):
            genre = self.GENRE_MAP[genre.lower()]
        if isinstance(genre, int):
            return torch.full((B,), genre, dtype=torch.long, device=device)
        if not isinstance(genre, torch.Tensor):
            return torch.tensor(genre, dtype=torch.long, device=device)
        return genre.to(device)

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        genre_ids: torch.Tensor,
        t: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Flow-matching MSE loss with optional variance-matching penalty.

        Args:
            x0:        (B, 1, n_mels, T) source mels
            x1:        (B, 1, n_mels, T) target mels
            genre_ids: (B,) target genre indices
            t:         (B,) optional; sampled uniformly in [0,1] if None
            mask:      (B, T) optional time mask (1=valid, 0=pad)

        Returns:
            scalar loss
        """
        x0, x1 = self._trim_to_same(x0, x1)
        B = x0.shape[0]
        device = x0.device

        if t is None:
            t = torch.rand(B, device=device)

        # Straight-path interpolation
        t4 = t.view(B, 1, 1, 1)
        x_t  = (1 - t4) * x0 + t4 * x1     # (B, 1, n_mels, T)
        v_t  = x1 - x0                       # target velocity

        v_pred = self.dit(x_t, t, genre_ids)  # (B, 1, n_mels, T_trim)

        # Align time dimension (patchify may trim trailing frames)
        T_out = v_pred.shape[-1]
        v_t = v_t[..., :T_out]
        if mask is not None:
            mask = mask[..., :T_out]

        loss_elem = F.mse_loss(v_pred, v_t, reduction="none")  # (B, 1, n_mels, T)

        if mask is not None:
            # mask: (B, T) → (B, 1, 1, T) for broadcasting
            mask4 = mask[:, None, None, :].float()
            n_valid = mask4.sum()
            mse_loss = (loss_elem * mask4).sum() / (n_valid + 1e-8)
        else:
            mse_loss = loss_elem.mean()

        # Variance-matching penalty over time axis (discourages lazy v≈0 solution)
        if self.std_penalty_weight > 0.0:
            std_penalty = F.mse_loss(
                v_pred.std(dim=-1),
                v_t.std(dim=-1),
            )
            return mse_loss + self.std_penalty_weight * std_penalty

        return mse_loss

    # ------------------------------------------------------------------
    # Samplers  (mirror models/flow.py exactly)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_euler(
        self,
        x0: torch.Tensor,
        genre_ids,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Euler integration."""
        B = x0.shape[0]
        device = x0.device
        genre_ids = self._genre_tensor(genre_ids, B, device)

        dt = 1.0 / num_steps
        xt = x0.clone()

        for step in range(num_steps):
            t = torch.full((B,), step / num_steps, device=device)
            v = self.dit(xt, t, genre_ids)
            T_out = v.shape[-1]
            xt = xt[..., :T_out] + v * dt

        return xt

    @torch.no_grad()
    def sample_heun(
        self,
        x0: torch.Tensor,
        genre_ids,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Heun's 2nd-order method. Halves integration error vs Euler."""
        B = x0.shape[0]
        device = x0.device
        genre_ids = self._genre_tensor(genre_ids, B, device)

        dt = 1.0 / num_steps
        xt = x0.clone()

        for step in range(num_steps):
            t      = torch.full((B,), step / num_steps,       device=device)
            t_next = torch.full((B,), (step + 1) / num_steps, device=device)

            v1 = self.dit(xt, t, genre_ids)
            T_out = v1.shape[-1]
            xt_euler = xt[..., :T_out] + v1 * dt

            v2 = self.dit(xt_euler, t_next, genre_ids)
            xt = xt[..., :T_out] + (v1 + v2) * (dt / 2)

        return xt

    def sample(
        self,
        x0: torch.Tensor,
        genre_ids,
        num_steps: int = 50,
        method: str = "euler",
    ) -> torch.Tensor:
        """
        Convenience wrapper — dispatches to the right sampler.

        Args:
            x0:        (B, 1, n_mels, T) source mel spectrogram
            genre_ids: int | str | (B,) long tensor  — target genre
            num_steps: ODE integration steps
            method:    "euler" | "heun"

        Returns:
            (B, 1, n_mels, T') transferred mel
        """
        if method == "euler":
            return self.sample_euler(x0, genre_ids, num_steps)
        elif method == "heun":
            return self.sample_heun(x0, genre_ids, num_steps)
        raise ValueError(f"Unknown sampling method: '{method}'. Use 'euler' or 'heun'.")

    # ------------------------------------------------------------------
    # forward  (training alias — matches Lightning / HuggingFace Trainer API)
    # ------------------------------------------------------------------

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        genre_ids: torch.Tensor,
        t: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.compute_loss(x0, x1, genre_ids, t, mask)
