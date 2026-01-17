# Mac Mini M4 Setup Guide - 3D Face Reconstruction

## Pre-Weekend Preparation

### 1. Download These Files Beforehand
Since you have Basel credentials, download these while on Windows:

**Basel Face Model 2009:**
- Visit: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads
- Login with your Basel credentials
- Download: `01_MorphableModel.mat` (~150-200 MB)
- Save to USB drive or cloud storage for transfer to Mac

**Test Images:**
- Already in: `c:\Users\00298204\Documents\Face3d\face3d_dev\images\bm\`
- Copy these 6 JPG files to transfer to Mac

### 2. Mac Mini M4 Prerequisites

Check these are installed (or will install on weekend):
- [ ] Xcode Command Line Tools: `xcode-select --install`
- [ ] Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- [ ] Python 3.10+: `brew install python@3.11`
- [ ] Git: `brew install git`

---

## Weekend Setup Workflow (Estimated 1-2 hours)

### Step 1: Clone Repository (5 min)
```bash
cd ~/Documents
git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch.git
cd Deep3DFaceRecon_pytorch
```

### Step 2: Create Virtual Environment (5 min)
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Step 3: Install PyTorch with MPS Support (10 min)
```bash
# PyTorch with Metal Performance Shaders for M4 GPU
pip install torch torchvision torchaudio
```

### Step 4: Install Dependencies (10 min)
```bash
pip install \
    numpy scipy scikit-image opencv-python pillow \
    kornia dominate tensorboard trimesh \
    dlib face-alignment
```

### Step 5: Clone InsightFace for ArcFace (5 min)
```bash
git clone --depth=1 https://github.com/deepinsight/insightface.git
cp -r insightface/recognition/arcface_torch models/
```

### Step 6: Install nvdiffrast (15 min)
```bash
# Install build tools first
pip install ninja

# Clone and install nvdiffrast
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
cd ..
```

**Note:** nvdiffrast on Mac will use OpenGL context (not Metal). M4 has excellent OpenGL 4.1 support.

### Step 7: Set Up BFM Folder (5 min)
```bash
# Copy the downloaded 01_MorphableModel.mat to BFM folder
cp /path/to/downloaded/01_MorphableModel.mat BFM/

# Download Expression Basis
cd BFM
curl -L -o Exp_Pca.bin "https://drive.google.com/uc?id=1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6&export=download"
cd ..

# Generate BFM_model_front.mat
python -c "from util.load_mats import transferBFM09; transferBFM09('BFM')"
```

### Step 8: Download Pre-trained Model (10 min)
```bash
pip install gdown
cd checkpoints
gdown --folder "https://drive.google.com/drive/folders/1liaIxn9smpudjjqMaWWRpP0mXRW_qRPP?usp=sharing"
cd ..
```

### Step 9: Fix numpy 2.x Compatibility (2 min)
Edit `util/preprocess.py` line 19:
```python
# Comment out this line:
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
```

Or run:
```bash
sed -i '' 's/warnings.filterwarnings.*VisibleDeprecationWarning.*/# &/' util/preprocess.py
```

### Step 10: Prepare Test Images (5 min)
```bash
mkdir -p datasets/test_faces/detections
cp /path/to/your/face/images/*.jpg datasets/test_faces/

# Generate 5-point landmarks using the detection script
# (Copy generate_detections.py from Windows work)
python generate_detections.py
```

### Step 11: Run First Test! (2-5 min)
```bash
# Use MPS backend (Metal GPU on M4)
export PYTORCH_ENABLE_MPS_FALLBACK=1

python test.py \
    --name=face_recon_feat0.2_augment \
    --epoch=20 \
    --img_folder=datasets/test_faces \
    --gpu_ids=0
```

**Expected Performance on M4:**
- First image: ~10-30 seconds (model loading + inference)
- Subsequent images: ~5-10 seconds each
- Much faster than CPU-only!

---

## Verification Checklist

After setup, verify these files exist:

**BFM Folder:**
- [x] `BFM/01_MorphableModel.mat`
- [x] `BFM/Exp_Pca.bin`
- [x] `BFM/BFM_model_front.mat` (generated)
- [x] `BFM/similarity_Lm3D_all.mat` (already in repo)

**Checkpoints:**
- [x] `checkpoints/face_recon_feat0.2_augment/epoch_20.pth`

**Test Data:**
- [x] `datasets/test_faces/*.jpg` (your images)
- [x] `datasets/test_faces/detections/*.txt` (5-point landmarks)

**Results After Test:**
- [x] `checkpoints/face_recon_feat0.2_augment/results/test_faces/*.obj` (3D meshes)
- [x] `checkpoints/face_recon_feat0.2_augment/results/test_faces/*.png` (visualizations)

---

## Troubleshooting

### If nvdiffrast fails to install:
```bash
# Install OpenGL dependencies
brew install glfw glew

# Try building with specific flags
MACOSX_DEPLOYMENT_TARGET=13.0 pip install git+https://github.com/NVlabs/nvdiffrast
```

### If MPS backend issues occur:
```bash
# Fallback to CPU (slower but works)
python test.py \
    --name=face_recon_feat0.2_augment \
    --epoch=20 \
    --img_folder=datasets/test_faces \
    --gpu_ids=-1
```

### If OpenGL context fails:
Add this flag: `--use_opengl False`

---

## Quality Evaluation Plan

Once you have results, we'll evaluate:

1. **Visual Quality** - Do the 3D meshes look realistic?
2. **Texture Quality** - Is the face texture mapped correctly?
3. **Speed** - Acceptable inference time for your use case?
4. **Comparison** - Test simpler alternatives (3DDFA_V2, face_alignment) for comparison

---

## Next Steps After Testing

**If quality is good:**
→ Plan production deployment to Linux server
→ Design web service architecture (Flask/FastAPI + job queue)
→ Set up Docker containers

**If quality needs improvement:**
→ Try fine-tuning on your specific dataset
→ Explore post-processing improvements
→ Consider ensemble methods

**If too slow even on M4:**
→ Optimize batch processing
→ Consider cloud GPU for production (AWS/GCP)
→ Implement smart caching strategies

---

## Files to Transfer from Windows to Mac

Create a folder with these:
1. `01_MorphableModel.mat` (downloaded from Basel)
2. Test face images from `face3d_dev/images/bm/`
3. `generate_detections.py` (the working version)
4. This setup guide

---

## Estimated Total Time

- **Setup**: 1-2 hours
- **First successful inference**: 2-3 hours (including troubleshooting)
- **Quality evaluation**: 30-60 min
- **Total weekend effort**: 3-5 hours

Good luck next weekend! The M4's Neural Engine and GPU should make this a much smoother experience than the Windows laptop.
