"""
Helper script to download Basel Face Model alternatives while waiting for official approval.
"""
import os
import urllib.request
import sys


def check_model_exists():
    """Check if Basel model is already downloaded."""
    model_path = "models/model2017-1_face12_nomouth.h5"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Basel Face Model found: {model_path}")
        print(f"  File size: {size_mb:.1f} MB")
        return True
    else:
        print(f"✗ Basel Face Model not found at: {model_path}")
        return False


def download_alternative_model():
    """
    Information about alternative face models that don't require registration.
    """
    print("\n" + "="*60)
    print("ALTERNATIVE 3D FACE MODELS")
    print("="*60)
    print("\nWhile waiting for Basel Face Model approval, you can use:")
    print("\n1. FLAME (Recommended Alternative)")
    print("   - Similar quality to Basel")
    print("   - Free for research")
    print("   - URL: https://flame.is.tue.mpg.de/")
    print("   - Download: FLAME 2020 model")

    print("\n2. FaceWarehouse")
    print("   - 150 high-quality 3D face scans")
    print("   - Includes expressions")
    print("   - URL: http://kunzhou.net/zjugaps/facewarehouse/")

    print("\n3. Surrey Face Model")
    print("   - Similar to Basel but different license")
    print("   - URL: https://cvssp.org/facemodel")

    print("\n" + "="*60)
    print("\nFor now, let me create a demo/test version...")
    print("="*60)


def create_models_directory():
    """Ensure models directory exists."""
    os.makedirs("models", exist_ok=True)
    print("✓ Created models/ directory")


def show_download_instructions():
    """Display clear download instructions."""
    print("\n" + "="*60)
    print("HOW TO DOWNLOAD BASEL FACE MODEL")
    print("="*60)

    print("\n1. Open your web browser and go to:")
    print("   https://faces.dmi.unibas.ch/bfm/bfm2017.html")

    print("\n2. Click 'Register' and fill out the form:")
    print("   - Name: Your name")
    print("   - Email: Your email address")
    print("   - Institution: University/Company (or 'Independent Researcher')")
    print("   - Purpose: 'Research and learning'")

    print("\n3. Check your email for approval (usually 2-6 hours)")

    print("\n4. After approval, log in and download:")
    print("   - File: model2017-1_face12_nomouth.h5")
    print("   - Size: ~280 MB")

    print("\n5. Move the downloaded file to:")
    print("   C:\\Users\\00298204\\Documents\\Face3d\\face3d_dev\\models\\")

    print("\n6. Run this script again to verify:")
    print("   python download_helper.py")

    print("\n" + "="*60)
    print("\nOr use PowerShell command:")
    print('   Move-Item "$env:USERPROFILE\\Downloads\\model2017-1_face12_nomouth.h5" "models\\"')
    print("="*60 + "\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BASEL FACE MODEL DOWNLOAD HELPER")
    print("="*60)

    create_models_directory()

    print("\nChecking for existing model...")
    if check_model_exists():
        print("\n✓ All set! You can now run:")
        print("  python basel_face_model.py")
    else:
        print("\nModel not found. Let's download it...")
        show_download_instructions()

        print("\nWould you like to see alternative models?")
        print("(These are available immediately without registration)")
        response = input("Show alternatives? (y/n): ").strip().lower()

        if response == 'y':
            download_alternative_model()

        print("\n" + "="*60)
        print("Once you've downloaded the model, run this script again!")
        print("="*60 + "\n")
