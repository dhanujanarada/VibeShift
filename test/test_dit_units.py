import unittest
import torch
import sys
sys.path.append('c:\\Users\\Dhanuja\\Desktop\\Vibeshift\\VibeShift')

from models.dit import DiT

class TestDiT(unittest.TestCase):
    
    def setUp(self):
        self.model = DiT(
            patch_dim=256,
            embed_dim=512,
            num_blocks=6,  # Smaller for testing
            num_heads=8,
            num_genres=5,
            hidden_dim=1024
        )
    
    def test_output_shape(self):
        """Test that output shape matches input"""
        x = torch.randn(4, 50, 256)
        t = torch.rand(4)
        genres = torch.randint(0, 5, (4,))
        
        output = self.model(x, t, genres)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_different_batch_sizes(self):
        """Test with various batch sizes"""
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 100, 256)
            t = torch.rand(batch_size)
            genres = torch.randint(0, 5, (batch_size,))
            
            output = self.model(x, t, genres)
            self.assertEqual(output.shape[0], batch_size)
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths"""
        for seq_len in [10, 50, 100, 200]:
            x = torch.randn(2, seq_len, 256)
            t = torch.rand(2)
            genres = torch.randint(0, 5, (2,))
            
            output = self.model(x, t, genres)
            self.assertEqual(output.shape[1], seq_len)
    
    def test_timestep_range(self):
        """Test with extreme timesteps"""
        x = torch.randn(2, 50, 256)
        genres = torch.randint(0, 5, (2,))
        
        for t in [0.0, 0.5, 1.0]:
            output = self.model(x, torch.tensor([t, t]), genres)
            self.assertFalse(torch.isnan(output).any())
    
    def test_all_genres(self):
        """Test that all genres are handled"""
        x = torch.randn(5, 50, 256)
        t = torch.rand(5)
        
        for genre_id in range(5):
            genres = torch.tensor([genre_id] * 5)
            output = self.model(x, t, genres)
            self.assertEqual(output.shape, x.shape)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model"""
        x = torch.randn(2, 50, 256, requires_grad=True)
        t = torch.rand(2)
        genres = torch.randint(0, 5, (2,))
        
        output = self.model(x, t, genres)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))

if __name__ == '__main__':
    unittest.main()