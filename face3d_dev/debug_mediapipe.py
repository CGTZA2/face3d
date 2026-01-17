import sys
import os

print(f"Python Executable: {sys.executable}")

try:
    import mediapipe as mp
    mp_loc = getattr(mp, '__file__', 'No file attribute')
    print(f"MediaPipe Location: {mp_loc}")
    
    # Check if the 'python' subfolder actually exists on disk
    if mp_loc and os.path.exists(mp_loc):
        mp_dir = os.path.dirname(mp_loc)
        print("Contents of mediapipe dir:", os.listdir(mp_dir))
        
    print(f"Has 'solutions'? {hasattr(mp, 'solutions')}")
    print(f"Dir(mp): {dir(mp)}")
except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"General Error: {e}")