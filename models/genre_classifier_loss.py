

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple


class GenreClassifierLoss(nn.Module):
    
    
    def __init__(
        self,
        num_genres: int = 2,
        mel_height: int = 100,
        mel_width: int = 512,
        embedding_dim: int = 128,
        loss_weight: float = 0.5,
        device: Optional[torch.device] = None,
        freeze_on_init: bool = True,
    ):
        super().__init__()
        
        self.num_genres = num_genres
        self.mel_height = mel_height
        self.mel_width = mel_width
        self.embedding_dim = embedding_dim
        self.loss_weight = loss_weight
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Build a simple genre classifier backbone
        self.classifier = self._build_classifier()
        
        # Add classification head for training
        self.classification_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_genres)
        )
        
        # Optionally freeze on initialization
        if freeze_on_init:
            self.freeze()
    
    def _build_classifier(self) -> nn.Module:
        
        return nn.Sequential(
            # Input: (B, 1, mel_height, mel_width)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (B, 32, mel_height//2, mel_width//2)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (B, 64, mel_height//4, mel_width//4)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 128, 1, 1)
        )
    
    def freeze(self):
        """Freeze all parameters in the classifier."""
        for param in self.classifier.parameters():
            param.requires_grad = False
        for param in self.classification_head.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all parameters in the classifier (for fine-tuning if needed)."""
        for param in self.classifier.parameters():
            param.requires_grad = True
        for param in self.classification_head.parameters():
            param.requires_grad = True
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract genre features from mel spectrogram.
        
        Args:
            x: (B, 1, mel_height, mel_width) or (B, C, mel_height, mel_width) mel spectrogram
        
        Returns:
            (B, embedding_dim) feature embeddings
        """
        # Ensure single channel input
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)  # Average channels if multi-channel
        
        with torch.no_grad():
            # Extract features using frozen classifier
            features = self.classifier(x)  # (B, 128, 1, 1)
            features = features.view(features.size(0), -1)  # (B, 128)
        
        return features
    
    def compute_genre_consistency_loss(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        target_genre_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute genre consistency loss using contrastive learning.
        
        This loss ensures that generated audio features are close to target audio features
        in the genre embedding space, making the style transfer more effective.
        
        Args:
            generated: (B, C, H, W) generated mel spectrogram
            target: (B, C, H, W) target genre mel spectrogram
            target_genre_id: (B,) target genre indices (1 for rock, 0 for non-rock)
        
        Returns:
            loss: scalar contrastive loss
        """
        batch_size = generated.size(0)
        device = generated.device
        
        # Extract features
        gen_features = self.extract_features(generated)      # (B, 128)
        target_features = self.extract_features(target)      # (B, 128)
        
        # Normalize features
        gen_features_norm = F.normalize(gen_features, dim=1)       # (B, 128)
        target_features_norm = F.normalize(target_features, dim=1)  # (B, 128)
        
        # Compute cosine similarity between generated and target
        # Higher similarity = generated is closer to target genre
        similarity = torch.matmul(gen_features_norm, target_features_norm.t())  # (B, B)
        
        # Create labels for contrastive loss (diagonal should be 1)
        labels = torch.eye(batch_size, device=device)
        
        # Contrastive loss (info-NCE style)
        # We want generated features to be similar to their corresponding target features
        loss = F.cross_entropy(similarity / 0.1, torch.arange(batch_size, device=device))
        
        return loss
    
    def compute_genre_alignment_loss(
        self,
        generated: torch.Tensor,
        target_genre_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute genre alignment loss to push generated features toward target genre.
        
        Uses a simple MSE loss to encourage generated features to match
        the learned representation of the target genre.
        
        Args:
            generated: (B, C, H, W) generated mel spectrogram
            target_genre_id: (B,) target genre indices
        
        Returns:
            loss: scalar alignment loss
        """
        # Extract features from generated audio
        gen_features = self.extract_features(generated)  # (B, 128)
        
        
        batch_size = gen_features.size(0)
        device = gen_features.device
    
        # Initialize as random directions (frozen)
        torch.manual_seed(42)  # For reproducibility
        genre_prototypes = F.normalize(
            torch.randn(self.num_genres, self.embedding_dim, device=device),
            dim=1
        )
        
        # Get prototypes for target genres
        target_prototypes = genre_prototypes[target_genre_id]  # (B, 128)
        
        # MSE loss between generated features and target genre prototype
        loss = F.mse_loss(gen_features, target_prototypes)
        
        return loss
    
    def train_classifier(
        self,
        train_loader,
        val_loader=None,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        save_path: Optional[str] = None,
    ):
        """
        Train the genre classifier.
        
        Args:
            train_loader: DataLoader with (mel_spec, genre_id) tuples
            val_loader: Optional validation DataLoader
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            save_path: Optional path to save best model
        """
        self.unfreeze()
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for mel_specs, genre_ids in train_loader:
                mel_specs = mel_specs.to(self.device)
                genre_ids = genre_ids.to(self.device)
                
                # Forward pass
                features = self.classifier(mel_specs)
                features = features.view(features.size(0), -1)
                logits = self.classification_head(features)
                
                loss = criterion(logits, genre_ids)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Metrics
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += genre_ids.size(0)
                train_correct += (predicted == genre_ids).sum().item()
            
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Validation
            if val_loader:
                val_acc = self.evaluate_classifier(val_loader)
                print(f"Val Acc: {val_acc:.2f}%")
                
                # Save best model
                if save_path and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_classifier(save_path)
                    print(f"✓ Saved best model with val_acc: {val_acc:.2f}%")
        
        # Freeze after training
        self.freeze()
        print("✓ Classifier training complete. Weights frozen.")
    
    def evaluate_classifier(self, data_loader) -> float:
        """
        Evaluate classifier accuracy.
        
        Args:
            data_loader: DataLoader with (mel_spec, genre_id) tuples
        
        Returns:
            accuracy: Accuracy percentage
        """
        self.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for mel_specs, genre_ids in data_loader:
                mel_specs = mel_specs.to(self.device)
                genre_ids = genre_ids.to(self.device)
                
                features = self.classifier(mel_specs)
                features = features.view(features.size(0), -1)
                logits = self.classification_head(features)
                
                _, predicted = torch.max(logits, 1)
                total += genre_ids.size(0)
                correct += (predicted == genre_ids).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def save_classifier(self, path: str):
        """Save classifier weights."""
        checkpoint = {
            'classifier_state': self.classifier.state_dict(),
            'classification_head_state': self.classification_head.state_dict(),
            'num_genres': self.num_genres,
            'embedding_dim': self.embedding_dim,
        }
        torch.save(checkpoint, path)
    
    def compute_loss(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        target_genre_id: torch.Tensor,
        use_consistency: bool = True,
        use_alignment: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined genre classifier loss.
        
        Args:
            generated: (B, C, H, W) generated mel spectrogram
            target: (B, C, H, W) target genre mel spectrogram
            target_genre_id: (B,) target genre indices
            use_consistency: Whether to use consistency loss
            use_alignment: Whether to use alignment loss
        
        Returns:
            loss: scalar combined loss
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        total_loss = 0.0
        
        if use_consistency:
            consistency_loss = self.compute_genre_consistency_loss(
                generated, target, target_genre_id
            )
            loss_dict['genre_consistency'] = consistency_loss.item()
            total_loss += consistency_loss
        
        if use_alignment:
            alignment_loss = self.compute_genre_alignment_loss(
                generated, target_genre_id
            )
            loss_dict['genre_alignment'] = alignment_loss.item()
            total_loss += alignment_loss
        
        # Apply loss weight
        final_loss = self.loss_weight * total_loss
        loss_dict['genre_classifier_loss'] = final_loss.item()
        
        return final_loss, loss_dict
    
    def load_pretrained(self, checkpoint_path: str) -> bool:
        """Load pretrained classifier weights and freeze."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                self.classifier.load_state_dict(checkpoint.get('classifier_state', checkpoint))
                if 'classification_head_state' in checkpoint:
                    self.classification_head.load_state_dict(checkpoint['classification_head_state'])
            else:
                self.classifier.load_state_dict(checkpoint)
            self.freeze()
            print(f"✓ Loaded pretrained classifier from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"⚠ Failed to load pretrained classifier: {e}")
            return False
    
    def to(self, device):
        """Override to() to move module to device."""
        self.device = device
        return super().to(device)


class GenreClassifierAuxiliaryLoss(nn.Module):
       
    def __init__(
        self,
        num_genres: int = 2,
        mel_height: int = 100,
        mel_width: int = 512,
        embedding_dim: int = 128,
        loss_weight: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.classifier_loss = GenreClassifierLoss(
            num_genres=num_genres,
            mel_height=mel_height,
            mel_width=mel_width,
            embedding_dim=embedding_dim,
            loss_weight=loss_weight,
            device=device,
        )
    
    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        target_genre_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute auxiliary loss.
        
        Args:
            generated: (B, C, H, W) generated mel spectrogram
            target: (B, C, H, W) target genre mel spectrogram
            target_genre_id: (B,) target genre indices
        
        Returns:
            loss: scalar loss
            loss_dict: Dictionary with loss components
        """
        return self.classifier_loss.compute_loss(
            generated, target, target_genre_id
        )
    
    def to(self, device):
        """Override to() to move module to device."""
        self.classifier_loss = self.classifier_loss.to(device)
        return super().to(device)
