import torch
from models.dit import DiT

class DiTTester:
    def __init__(
        self,
        model,
        patch_dim=256,
        seq_len=16,
        num_genres=10,
        device="cpu",
    ):
        self.model = model.to(device)
        self.patch_dim = patch_dim
        self.seq_len = seq_len
        self.num_genres = num_genres
        self.device = device

    def _dummy_inputs(self, batch_size=4, requires_grad=False):
        x = torch.randn(
            batch_size,
            self.seq_len,
            self.patch_dim,
            device=self.device,
            requires_grad=requires_grad,
        )
        t = torch.rand(batch_size, device=self.device)
        genre_ids = torch.randint(
            0, self.num_genres, (batch_size,), device=self.device
        )
        return x, t, genre_ids

    def test_forward(self):
        print("▶ Forward pass test")
        x, t, genre_ids = self._dummy_inputs()

        with torch.no_grad():
            y = self.model(x, t, genre_ids)

        assert y.shape == x.shape, f"Output shape {y.shape} != input {x.shape}"
        print("  ✓ Forward pass OK")

    def test_gradients(self):
        print("▶ Gradient flow test")
        x, t, genre_ids = self._dummy_inputs(requires_grad=True)

        y = self.model(x, t, genre_ids)
        loss = y.mean()
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

        print("  ✓ Gradients OK")

    def test_conditioning_effect(self):
        print("▶ Conditioning influence test")
        x, _, _ = self._dummy_inputs()

        t0 = torch.zeros(x.size(0), device=self.device)
        t1 = torch.ones(x.size(0), device=self.device)

        g0 = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        g1 = torch.ones(x.size(0), dtype=torch.long, device=self.device)

        y_t0 = self.model(x, t0, g0)
        y_t1 = self.model(x, t1, g0)
        y_g1 = self.model(x, t0, g1)

        t_diff = (y_t0 - y_t1).abs().mean().item()
        g_diff = (y_t0 - y_g1).abs().mean().item()

        assert t_diff > 1e-5, "Timestep conditioning has no effect"
        assert g_diff > 1e-5, "Genre conditioning has no effect"

        print(f"  ✓ Timestep diff: {t_diff:.4f}")
        print(f"  ✓ Genre diff:    {g_diff:.4f}")

    def test_overfit_tiny_batch(self, steps=200, lr=1e-3):
        print("▶ Tiny batch overfitting test")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        x, t, genre_ids = self._dummy_inputs(batch_size=2)
        target = torch.randn_like(x)

        initial_loss = None
        for step in range(steps):
            optimizer.zero_grad()
            y = self.model(x, t, genre_ids)
            loss = ((y - target) ** 2).mean()

            if step == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.5, "Model failed to overfit tiny batch"

        print(f"  ✓ Loss {initial_loss:.4f} → {final_loss:.4f}")

    def test_output_statistics(self):
        print("▶ Output statistics test")
        x, t, genre_ids = self._dummy_inputs()

        y = self.model(x, t, genre_ids)

        mean = y.mean().item()
        std = y.std().item()

        assert not torch.isnan(y).any(), "NaNs detected in output"
        assert std > 0, "Output variance collapsed"

        print(f"  ✓ Mean: {mean:.4f}, Std: {std:.4f}")

    def run_all(self):
        print("\n=== DiT Test Suite ===")
        self.test_forward()
        self.test_gradients()
        self.test_conditioning_effect()
        self.test_output_statistics()
        self.test_overfit_tiny_batch()
        print("=== All tests passed ===\n")

if __name__ == '__main__':
    model = DiT(
    patch_dim=256,
    embed_dim=512,
    num_blocks=2,  # keep small for tests
    num_heads=8,
    num_genres=10,)

    tester = DiTTester(model)
    tester.run_all()