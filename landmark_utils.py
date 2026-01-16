import cv2
import mediapipe as mp
import numpy as np
import sys
import os

class LandmarkDetector:
    def __init__(self, min_confidence=0.5, refine_landmarks=True):
        # Check for common shadowing issue (e.g. local mediapipe.py file)
        if not hasattr(mp, 'solutions'):
            try:
                # Attempt to force load the submodule to reveal the real error
                import mediapipe.python.solutions as solutions
                mp.solutions = solutions
            except Exception as e:
                # PRIORITY CHECK: Python Version Compatibility
                if sys.version_info >= (3, 13):
                     raise ImportError(
                        f"UNSUPPORTED PYTHON VERSION: {sys.version.split()[0]}\n"
                        "MediaPipe currently does not support Python 3.13 or newer.\n"
                        "ACTION: Please install Python 3.11 or 3.12 from python.org.\n"
                        "Then recreate your venv: 'py -3.11 -m venv .venv'"
                    )

                # DEBUG: Check if mediapipe is loaded from the current directory (Shadowing)
                mp_file = getattr(mp, '__file__', None)
                mp_path = getattr(mp, '__path__', [])
                cwd = os.getcwd()
                
                # Check if the loaded path is inside the current working directory
                is_local = False
                location = "unknown"
                if mp_file and cwd in os.path.abspath(mp_file) and "site-packages" not in mp_file:
                    is_local = True
                    location = mp_file
                elif mp_path:
                    for p in mp_path:
                        if cwd in os.path.abspath(p) and "site-packages" not in p:
                            is_local = True
                            location = p
                            break
                
                # DEBUG: Check if the installation is valid (contains 'python' folder)
                if not is_local and mp_file:
                    mp_dir = os.path.dirname(mp_file)
                    python_sub_dir = os.path.join(mp_dir, 'python')
                    if not os.path.exists(python_sub_dir):
                         raise ImportError(
                            f"CORRUPTED INSTALLATION DETECTED.\n"
                            f"Python Version: {sys.version.split()[0]}\n"
                            f"MediaPipe Path: {mp_dir}\n"
                            f"Missing Folder: {python_sub_dir}\n"
                            "CAUSE: The 'mediapipe' package installed by pip is incomplete.\n"
                            "FIX: Run these commands in PowerShell:\n"
                            "1. deactivate\n"
                            "2. Remove-Item -Recurse -Force .venv\n"
                            "3. pip cache purge\n"
                            "4. py -3.11 -m venv .venv\n"
                            "5. .\\.venv\\Scripts\\Activate\n"
                            "6. python -m pip install --no-cache-dir -r requirements.txt"
                        )

                if is_local:
                     raise ImportError(
                        f"CONFLICT DETECTED: Python is loading 'mediapipe' from '{location}'\n"
                        "Instead of the installed library in site-packages.\n"
                        "ACTION: Please DELETE or RENAME this file/folder and try again."
                    )

                if "mediapipe.python" in str(e):
                    fix_cmd = "pip install -r requirements.txt --force-reinstall"
                    cause = "The MediaPipe installation is corrupted (missing submodules)."
                else:
                    fix_cmd = "pip install --upgrade protobuf"
                    cause = "Your 'protobuf' library might be incompatible."

                raise ImportError(
                    f"MediaPipe is installed but failed to load components.\n"
                    f"Underlying Error: {e}\n"
                    f"Cause: {cause}\n"
                    f"Try running:\n{fix_cmd}"
                )

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,       # We are processing static images, not a video stream
            max_num_faces=1,              # We expect one face per image
            refine_landmarks=refine_landmarks, # detailed eyes and lips
            min_detection_confidence=min_confidence
        )

    def get_landmarks(self, image_bgr):
        """
        Detects landmarks in a BGR image (OpenCV format).
        Returns np.array of (x, y) points or None if no face detected.
        """
        if image_bgr is None:
            return None
            
        h, w, _ = image_bgr.shape
        
        # MediaPipe expects RGB, OpenCV gives BGR
        results = self.face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None
            
        # Convert normalized coordinates (0.0-1.0) to pixel coordinates
        landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks])
        return coords