# Basel Face Model Setup Guide

## Step 1: Download the Model

### Option A: BFM 2017 (Recommended - Includes Expression Model)
1. Go to: https://faces.dmi.unibas.ch/bfm/bfm2017.html
2. Register for an account (free for research/academic use)
3. Download: **model2017-1_face12_nomouth.h5** (~280 MB)
4. Place it in: `face3d_dev/models/`

### Option B: BFM 2009 (Simpler, No Expression)
1. Go to: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads
2. Register and download the model
3. Place files in: `face3d_dev/models/`

## Step 2: Install Required Packages

Run these commands in your activated venv:

```bash
pip install h5py
pip install chumpy
pip install scikit-image
```

## Step 3: Model File Structure

After download, your directory should look like:
```
face3d_dev/
  ├── models/
  │   └── model2017-1_face12_nomouth.h5  (or BFM 2009 files)
  ├── images/
  ├── output/
  └── [python scripts]
```

## Why Basel Face Model?

**Advantages:**
- High quality 3D mesh (~53K vertices vs 468 landmarks)
- Statistical model based on 200+ scanned faces
- Can generate any realistic face shape
- Proper texture UV mapping
- View from any angle with good results

**How it works:**
1. Start with average face shape
2. Adjust shape parameters to match your landmarks
3. Fit expression parameters (optional)
4. Project multi-view textures onto the model
5. Result: Full 3D head model viewable from all angles

## Next Steps

Once you've downloaded the model:
1. Place it in the `models/` folder
2. Run the Basel fitting script (I'll create this next)
3. View your high-quality 3D reconstruction

---

**Note:** The registration is free but may take a few hours for approval.
If you want to start immediately, let me know and I can show you alternative models
or a demo with synthetic data.
