import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from models.dit import DiT
from models.flow import FlowMatching
from training.dataloader import RandomPairMelDataset


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
	patch_height: int = 10
	patch_width: int = 4
	embed_dim: int = 256
	num_blocks: int = 4
	num_heads: int = 4
	hidden_dim: int = 1024
	dropout: float = 0.1
	num_genres: int = 2
	in_channels: int = 1


class TrainingPipeline:
	def __init__(self, config: TrainingConfig):
		self.config = config
		self.device = torch.device(config.device)

		# Lightweight DiT configuration
		self.dit = DiT(
			in_channels=self.config.in_channels,
			patch_height=self.config.patch_height,
			patch_width=self.config.patch_width,
			embed_dim=self.config.embed_dim,
			num_blocks=self.config.num_blocks,
			num_heads=self.config.num_heads,
			hidden_dim=self.config.hidden_dim,
			num_genres=self.config.num_genres,
			dropout=self.config.dropout,
		).to(self.device)

		self.flow = FlowMatching(
			self.dit,
			use_genre_loss=True,
			genre_loss_weight=0.5,
			num_genres=self.config.num_genres,
			mel_height=100,
			mel_width=512,
		).to(self.device)
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
		x0_list, x1_list = zip(*batch)
		max_time = max(x.shape[-1] for x in x0_list)
		if self.config.max_time is not None:
			max_time = min(max_time, self.config.max_time)

		def pad(x: torch.Tensor):
			if x.shape[-1] > max_time:
				return x[..., :max_time]
			pad_amt = max_time - x.shape[-1]
			if pad_amt == 0:
				return x
			return torch.nn.functional.pad(x, (0, pad_amt))

		x0 = torch.stack([pad(x) for x in x0_list])
		x1 = torch.stack([pad(x) for x in x1_list])

		mask = torch.zeros((len(batch), 1, 1, max_time))
		for i, x in enumerate(x0_list):
			mask[i, :, :, : x.shape[-1]] = 1

		return x0, x1, mask

	def setup_data(self, source_files: List[str], target_files: List[str]) -> DataLoader:
		dataset = RandomPairMelDataset(source_files, target_files)
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
				x0 = x0 * mask
				x1 = x1 * mask

				genre_ids = torch.ones(x0.size(0), device=self.device, dtype=torch.long)

			loss, loss_dict = self.flow.compute_loss(x0, x1, genre_ids)

			self.optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.flow.parameters(), self.config.grad_clip_norm)
			self.optimizer.step()

			epoch_loss += loss.item()

			if step % 10 == 0 or step == len(loader):
				loss_str = f"Loss: {loss.item():.4f}"
				if 'flow_matching_loss' in loss_dict:
					loss_str += f" | Flow: {loss_dict['flow_matching_loss']:.4f}"
				if 'genre_classifier_loss' in loss_dict:
					loss_str += f" | Genre: {loss_dict['genre_classifier_loss']:.4f}"
				print(f"Epoch {epoch}/{self.config.num_epochs} - Step {step}/{len(loader)} - {loss_str}")
			self.scheduler.step(avg_loss)
			
		# Check if best model and save immediately
		is_best = avg_loss < best_loss
		if is_best:
			best_loss = avg_loss
			print(f"🌟 New best loss: {best_loss:.4f}")
			# Save best model immediately
			best_path = self.checkpoint_dir / "best_model.pt"
			state = {
				"epoch": epoch,
				"avg_loss": avg_loss,
				"best_loss": best_loss,
				"model_state": self.flow.state_dict(),
				"optimizer_state": self.optimizer.state_dict(),
				"scheduler_state": self.scheduler.state_dict(),
				"config": self.config.__dict__,
			}
			torch.save(state, best_path)
			print(f"✓ Saved best model instantly: {best_path}")
		
		# Save regular checkpoints at specified intervals
		if epoch % self.config.checkpoint_interval == 0:
			print(f"Saving checkpoint for epoch {epoch}...")
			ckpt = self.save_checkpoint(
				epoch,
				avg_loss,
				is_best=False,  # Already saved above if best
				best_loss=best_loss,
			)
			print(f"Checkpoint saved for epoch {epoch}: {ckpt}")

		return losses


def build_pipeline() -> TrainingPipeline:
	config = TrainingConfig()
	return TrainingPipeline(config)


if __name__ == "__main__":
	print("Training pipeline module. Import TrainingPipeline and TrainingConfig to use.")
