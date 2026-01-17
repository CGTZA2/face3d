# Download Basel Face Model 2009

Deep3DFaceRecon requires the Basel Face Model 2009 to work. Here's how to download it:

## Step 1: Access the Basel Face Model Website

Visit: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads

## Step 2: Log in with Your Basel Credentials

You already have Basel credentials that worked for BFM 2019. Use the same credentials:
- **Username**: (the one you used before)
- **Password**: (the one you used before)

## Step 3: Download the Required File

After logging in, download:
- **01_MorphableModel.mat** (approximately 150-200 MB)

## Step 4: Place the File

Move the downloaded file to:
```
c:\Users\00298204\Documents\Face3d\Deep3DFaceRecon_pytorch\BFM\01_MorphableModel.mat
```

## Step 5: Generate BFM_model_front.mat

After downloading, run this Python script to generate the required model file:

```bash
cd c:/Users/00298204/Documents/Face3d/Deep3DFaceRecon_pytorch
python -c "from util.load_mats import transferBFM09; transferBFM09('BFM')"
```

This will create `BFM/BFM_model_front.mat` which is needed for reconstruction.

## Verification

After completing these steps, you should have these files in the BFM folder:
- [x] Exp_Pca.bin (already downloaded - 51.1 MB)
- [x] std_exp.txt (already exists)
- [x] similarity_Lm3D_all.mat (already exists)
- [ ] 01_MorphableModel.mat (needs manual download)
- [ ] BFM_model_front.mat (will be generated from 01_MorphableModel.mat)

## Alternative: Try Without BFM09 First

We can attempt to run inference without BFM09 to see exactly what error we get. This might reveal if there's a workaround or if the model can run with partial functionality.

Try running:
```bash
cd c:/Users/00298204/Documents/Face3d/Deep3DFaceRecon_pytorch
python test.py --name=face_recon_feat0.2_augment --epoch=20 --img_folder=datasets/test_faces --gpu_ids=-1
```

If it fails with a specific error about BFM_model_front.mat, then the manual download is definitely required.
