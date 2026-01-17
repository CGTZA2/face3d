# Deep3DFaceRecon_pytorch Setup Guide

## System Compatibility

✅ **Your System:** Ryzen 7, 8GB RAM, Windows 11
- Can run on CPU (30-90 seconds per face)
- Enough RAM for testing
- Good for development and quality evaluation

## Installation Steps (CPU-Only)

### 1. Clone Repository

```bash
cd c:/Users/00298204/Documents/Face3d
git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch.git
cd Deep3DFaceRecon_pytorch
```

### 2. Install Dependencies

Option A: Use your existing venv
```bash
cd ../
.venv/Scripts/activate
cd Deep3DFaceRecon_pytorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Option B: Create new conda environment (if you have conda)
```bash
conda env create -f environment.yml
conda activate deep3d_pytorch
```

### 3. Download Required Models

You need several pre-trained models:

**Basel Face Model 2009:**
- You already have Basel 2019
- They use BFM09 - might work with conversion
- Download from: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads

**Pre-trained Network Weights:**
```bash
# Download from their Google Drive or GitHub releases
# epoch_20.pth (main model weights)
# Place in: checkpoints/pretrained/
```

**Face Recognition Model (optional):**
```bash
# InsightFace model for face recognition loss
# Can skip for basic inference
```

### 4. Prepare Your Face Images

```bash
# Create test folder
mkdir testdata/examples
# Copy your face images
cp ../face3d_dev/images/bm/*_frontal.jpg testdata/examples/
```

### 5. Run Inference (CPU Mode)

Edit `test.py` to force CPU:
```python
# Add at top of test.py
import torch
torch.set_default_tensor_type('torch.FloatTensor')  # Force CPU
device = torch.device('cpu')
```

Run:
```bash
python test.py --name=pretrained --img_folder=testdata/examples
```

## Expected Results

**Output Files (per face):**
- `*_shape.obj` - 3D mesh geometry
- `*_texture.png` - UV texture map
- `*_coeffs.txt` - Model parameters
- `*_depth.png` - Depth map

**Quality:**
- Better than MediaPipe (468 vertices)
- Better than basic Basel fitting
- State-of-the-art facial reconstruction
- Proper handling of pose/lighting

## Limitations on CPU

⚠️ **Slow Processing:**
- First image: ~2 minutes (model loading)
- Subsequent images: ~30-90 seconds each
- Total for 2 faces: ~3-5 minutes

⚠️ **Memory:**
- Needs ~4-6GB RAM during processing
- Close other apps if you have issues

## Alternative: Simplified Test

If full setup is too complex, try **face3d library** instead:

```bash
pip install face3d-py
```

This is lighter weight but may have lower quality.

## Production Deployment Strategy

After testing on laptop:

1. **If quality is good** → Deploy to Linux server
2. **CPU server** ($20-50/mo):
   - Use Celery + Redis for job queue
   - Process in background (30-60s per face)
   - Email notification when done

3. **GPU server** ($100-200/mo):
   - Real-time processing (3-5 seconds)
   - Better user experience
   - AWS/GCP GPU instances

## Troubleshooting

**Out of Memory:**
```python
# Reduce batch size in config
# Process one image at a time
```

**CUDA Errors (even on CPU):**
```python
# Ensure you installed CPU version of PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Missing Dependencies:**
```bash
# Install one by one
pip install numpy opencv-python scipy scikit-image
pip install trimesh pymeshlab
```

## Resources

- **GitHub**: https://github.com/sicxu/Deep3DFaceRecon_pytorch
- **Paper**: "Accurate 3D Face Reconstruction with Weakly-Supervised Learning"
- **PyTorch CPU Installation**: https://huggingface.co/blog/daya-shankar/pytorch-install-guide
- **System Requirements**: https://www.geeksforgeeks.org/python/pytorch-system-requirements/

## Quick Decision

**Test on laptop first:**
1. Clone repo → 5 minutes
2. Install deps → 10 minutes
3. Download models → 15 minutes
4. Test on 2 faces → 5 minutes
5. Evaluate quality → You decide!

**Total time:** ~30-40 minutes to see if it works for you

If quality is good, proceed with server deployment.
If not, we explore other options.

## Next Steps

1. Try the installation
2. Process your 2 test faces
3. Check if quality meets your needs
4. Report back - we'll proceed based on results!
