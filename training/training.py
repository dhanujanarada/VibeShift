import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from models.dit import DiT
from models.flow import FlowMatching
from training.dataloader import DACDataset


@dataclass
class TrainingConfig:
	num_epochs: int = 10
	batch_size: int = 2
	learning_rate: float = 1e-4
	weight_decay: float = 1e-4
	grad_clip_norm: float = 1.0
	checkpoint_interval: int = 1
	checkpoint_dir: str = "checkpoints"
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	max_time: Optional[int] = None
	# DAC / DiT configuration
	input_dim: int = 768  # DAC latent dimension
	embed_dim: int = 256
	num_blocks: int = 4
	num_heads: int = 4
	hidden_dim: int = 1024
	dropout: float = 0.1
	num_genres: int = 3  # 0=classical, 1=rock, 2=unknown


class TrainingPipeline:
	def __init__(self, config: TrainingConfig):
		self.config = config
		self.device = torch.device(config.device)

		# DiT configuration for DAC embeddings
		self.dit = DiT(
			input_dim=self.config.input_dim,  # DAC latent dim (768)
			embed_dim=self.config.embed_dim,
			num_blocks=self.config.num_blocks,
			num_heads=self.config.num_heads,
			hidden_dim=self.config.hidden_dim,
			num_genres=self.config.num_genres,
			dropout=self.config.dropout,
		).to(self.device)

		self.flow = FlowMatching(self.dit).to(self.device)
		self.optimizer = torch.optim.AdamW(
			self.flow.parameters(),
			lr=self.config.learning_rate,
			weight_decay=self.config.weight_decay,
		)

		# Add scheduler after optimizer
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			self.optimizer,
			mode='min',
			factor=0.5,
			patience=3
		)

		self.checkpoint_dir = Path(self.config.checkpoint_dir)
		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

	def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]):
		"""Collate DAC embeddings (T, latent_dim) into batched tensors."""
		x0_list, x1_list = zip(*batch)
		max_time = max(x.shape[0] for x in x0_list)  # T dimension is first
		if self.config.max_time is not None:
			max_time = min(max_time, self.config.max_time)

		def pad(x: torch.Tensor):
			# x is (T, D), pad along T dimension
			if x.shape[0] > max_time:
				return x[:max_time, :]
			pad_amt = max_time - x.shape[0]
			if pad_amt == 0:
				return x
			# Pad (T, D) -> (max_time, D)
			return torch.nn.functional.pad(x, (0, 0, 0, pad_amt))

		x0 = torch.stack([pad(x) for x in x0_list])  # (B, max_time, D)
		x1 = torch.stack([pad(x) for x in x1_list])  # (B, max_time, D)

		# Mask shape: (B, max_time) for DAC
		mask = torch.zeros((len(batch), max_time))
		for i, x in enumerate(x0_list):
			mask[i, :x.shape[0]] = 1

		return x0, x1, mask

	def setup_data(self, source_files: List[str], target_files: List[str]) -> DataLoader:
		dataset = DACDataset(source_files, target_files)
		loader = DataLoader(
			dataset,
			batch_size=self.config.batch_size,
			shuffle=True,
			num_workers=0,
			collate_fn=self.collate_fn,
		)
		return loader

	def save_checkpoint(
		self,
		epoch: int,
		avg_loss: float,
		is_best: bool = False,
		best_loss: Optional[float] = None,
	) -> str:
		ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
		state = {
			"epoch": epoch,
			"avg_loss": avg_loss,
			"best_loss": best_loss if best_loss is not None else avg_loss,
			"model_state": self.flow.state_dict(),
			"optimizer_state": self.optimizer.state_dict(),
			"scheduler_state": self.scheduler.state_dict(),  # Save scheduler
			"config": self.config.__dict__,
		}
		torch.save(state, ckpt_path)
		
		if is_best:
			best_path = self.checkpoint_dir / "best_model.pt"
			torch.save(state, best_path)
			print(f"✓ Saved best model: loss={avg_loss:.4f}")
		
		return str(ckpt_path)

	def load_checkpoint(self, ckpt_path: str):
		checkpoint = torch.load(ckpt_path, map_location=self.device)
		self.flow.load_state_dict(checkpoint['model_state'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state'])
		self.scheduler.load_state_dict(checkpoint['scheduler_state'])  # Restore scheduler
		return checkpoint

	def train(
		self,
		loader: DataLoader,
		start_epoch: int = 1,
		end_epoch: Optional[int] = None,
		best_loss: Optional[float] = None,
	) -> List[float]:
		self.flow.train()
		losses: List[float] = []
		best_loss = best_loss if best_loss is not None else float('inf')
		printed_shapes = False

		if end_epoch is None:
			end_epoch = self.config.num_epochs
		if start_epoch > end_epoch:
			return losses

		for epoch in range(start_epoch, end_epoch + 1):
			print(f"\nEpoch {epoch}/{self.config.num_epochs} started")
			epoch_loss = 0.0
			for step, (x0, x1, mask) in enumerate(loader, start=1):
				x0 = x0.to(self.device)
				x1 = x1.to(self.device)
				mask = mask.to(self.device)

				if not printed_shapes:
					print("Starting batch shapes:")
					print(f"  x0: {x0.shape}")
					print(f"  x1: {x1.shape}")
					print(f"  mask: {mask.shape}")
					printed_shapes = True

				# Apply mask to ignore padded regions
				# mask is (B, T), expand to (B, T, 1) for broadcasting with (B, T, D)
				mask_expanded = mask.unsqueeze(-1)
				x0 = x0 * mask_expanded
				x1 = x1 * mask_expanded

				genre_ids = torch.ones(x0.size(0), device=self.device, dtype=torch.long)

				loss = self.flow.compute_loss(x0, x1, genre_ids)

				self.optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.config.grad_clip_norm)
				self.optimizer.step()

				epoch_loss += loss.item()

				if step % 10 == 0 or step == len(loader):
					print(f"Epoch {epoch}/{self.config.num_epochs} - Step {step}/{len(loader)} - Loss: {loss.item():.4f}")

			avg_loss = epoch_loss / max(1, len(loader))
			losses.append(avg_loss)
			print(f"Epoch {epoch}/{self.config.num_epochs} done - Avg Loss: {avg_loss:.4f}")
			
			# Update scheduler
			self.scheduler.step(avg_loss)
			
			# Check if best model
			is_best = avg_loss < best_loss
			if is_best:
				best_loss = avg_loss
			
			if epoch % self.config.checkpoint_interval == 0:
				print(f"Saving checkpoint for epoch {epoch}...")
				ckpt = self.save_checkpoint(
					epoch,
					avg_loss,
					is_best=is_best,
					best_loss=best_loss,
				)
				print(f"Checkpoint saved for epoch {epoch}: {ckpt}")

		return losses


def build_pipeline() -> TrainingPipeline:
	config = TrainingConfig()
	return TrainingPipeline(config)


if __name__ == "__main__":
	print("Training pipeline module. Import TrainingPipeline and TrainingConfig to use.")
