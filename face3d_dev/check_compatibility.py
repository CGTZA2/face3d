import sys
import platform

def check():
    print(f"--- Environment Check ---")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    if sys.version_info >= (3, 13):
        print("\n[CRITICAL WARNING]")
        print("You are using Python 3.13+. MediaPipe currently ONLY supports up to Python 3.12.")
        print("You MUST install Python 3.11 or 3.12 to run this project.")
        return

    try:
        import mediapipe
        print(f"\nMediaPipe found at: {mediapipe.__file__}")
    except ImportError:
        print("\nMediaPipe not found.")

if __name__ == "__main__":
    check()