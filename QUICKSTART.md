# VibeShift - Quick Start Guide

## 🚀 Getting Started

### Step 1: Test Your Setup
Before running the application, test that everything is properly configured:

```bash
python test_pipeline.py
```

This will verify:
- ✅ Checkpoint file exists
- ✅ All dependencies are installed
- ✅ Model modules can be imported
- ✅ Configuration files are present

### Step 2: Start the Application

```bash
python run_app.py
```

The server will start at: **http://127.0.0.1:8000**

### Step 3: Transform Your Audio

1. **Open your browser** to `http://127.0.0.1:8000`
2. **Upload an audio file** (supports WAV, MP3, FLAC)
3. **Click "Transform to Rock"**
4. **Wait** for the transformation (typically 30-60 seconds)
5. **Listen** to the preview in your browser
6. **Download** your transformed audio!

## 🎵 Supported Audio Formats

- **WAV** (recommended)
- **MP3**
- **FLAC**

## ⚡ Performance Tips

### For Faster Inference:
- Use a CUDA-capable GPU
- Keep audio files under 30 seconds for quick testing
- Reduce `num_steps` in the inference config (trade-off: quality vs speed)

### For Better Quality:
- Use high-quality input audio (44.1kHz or higher)
- Increase `num_steps` to 100 for smoother transformations
- Use WAV format to avoid compression artifacts

## 🔧 Configuration

### Checkpoint Path
The default checkpoint location is:
```
notebooks/checkpoints/paired_training/best_model.pt
```

To use a different checkpoint, edit [app/routes/frontend.py](app/routes/frontend.py):
```python
CHECKPOINT_PATH = ROOT_DIR / "path" / "to" / "your" / "checkpoint.pt"
```

### Inference Parameters
Edit [inference.py](inference.py) to modify:
- `target_genre`: 0=classical, 1=rock (default), 2=unknown
- `num_steps`: Number of flow matching steps (default: 50)
- `method`: Sampling method - "euler" or "heun" (default: "heun")

### Model Configuration
Edit [configs/dit.yaml](configs/dit.yaml) for model architecture settings.

## 📁 Output Files

Transformed audio files are saved in:
```
app/outputs/
```

Files are automatically named with unique IDs:
```
<uuid>_rock.wav
```

## 🐛 Troubleshooting

### Server won't start
```bash
# Check if port 8000 is already in use
# On Windows PowerShell:
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess

# Kill the process if needed
Stop-Process -Id <PID>
```

### Model loading errors
- Ensure checkpoint exists at the correct path
- Verify checkpoint is not corrupted (check file size > 0)
- Try loading checkpoint manually with PyTorch:
  ```python
  import torch
  checkpoint = torch.load("path/to/checkpoint.pt")
  print(checkpoint.keys())
  ```

### Out of memory errors
- Use CPU instead of GPU:
  Edit `inference.py` and change:
  ```python
  device: str = "cpu"
  ```
- Process shorter audio clips
- Close other applications using GPU/RAM

### Audio quality issues
- Check input audio quality
- Increase `num_steps` parameter
- Try different sampling methods

## 💡 Tips & Tricks

1. **Best Results**: Use instrumental music for cleaner transformations
2. **Batch Processing**: You can process multiple files by uploading them one after another
3. **Custom Genres**: Train your own genre transformations and swap the checkpoint
4. **Integration**: The API can be called programmatically - see the inference module

## 📊 Expected Performance

- **Loading model**: 5-15 seconds (first time only)
- **Transforming 10s audio**: ~30 seconds on GPU, ~2 minutes on CPU
- **Transforming 30s audio**: ~60 seconds on GPU, ~5 minutes on CPU

## 🎯 Next Steps

- Experiment with different audio files
- Try adjusting the transformation parameters
- Train custom genre models
- Integrate with your own applications

## 🆘 Need Help?

Check the main [README](README.md) for more detailed information about the architecture and implementation.

---

Happy transforming! 🎸🎵
