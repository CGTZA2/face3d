import os
import re
import glob
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FaceTriplet:
    """
    Represents a single subject with three specific camera views.
    """
    subject_id: str
    frontal_path: Optional[str] = None
    three_quarter_path: Optional[str] = None
    profile_path: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Returns True only if all three required views are present."""
        return all([self.frontal_path, self.three_quarter_path, self.profile_path])

    def get_paths(self) -> Tuple[str, str, str]:
        """Returns the paths in order: (Frontal, 3/4, Profile). Raises error if incomplete."""
        if not self.is_complete:
            raise ValueError(f"Triplet for subject {self.subject_id} is incomplete.")
        return (self.frontal_path, self.three_quarter_path, self.profile_path)

class ImageManager:
    def __init__(self, image_dir: str = "images/bm"):
        self.image_dir = image_dir
        self.triplets: Dict[str, FaceTriplet] = {}
        self.supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

    def _parse_filename(self, filepath: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parses the filename to extract Subject ID and View.
        
        Convention based on user input:
        - nbmf001.jpg  -> Frontal (f), ID: nbm001
        - nbmf0042.jpg -> Frontal (f), ID: nbm0042
        
        Returns:
            (subject_id, view_char) or (None, None) if parsing fails.
        """
        filename = os.path.basename(filepath)
        name, _ = os.path.splitext(filename)

        # Regex to match: (Prefix)(ViewChar)(Digits)
        # This handles variable length digits (e.g. 001 vs 0042)
        match = re.match(r"^(.*)([ftp])(\d+)$", name.lower())
        
        if match:
            prefix, view_char, digits = match.groups()
            subject_id = f"{prefix}{digits}"
            return subject_id, view_char
        
        logger.warning(f"File {filename}: does not match pattern [prefix][f|t|p][digits]. Skipping.")
        return None, None

    def scan_directory(self) -> None:
        """Scans the configured directory and groups images into triplets."""
        if not os.path.exists(self.image_dir):
            logger.error(f"Directory not found: {self.image_dir}")
            return

        logger.info(f"Scanning directory: {self.image_dir}")
        
        files = []
        for ext in self.supported_extensions:
            # Recursive search can be enabled by adding recursive=True to glob
            files.extend(glob.glob(os.path.join(self.image_dir, ext)))

        if not files:
            logger.warning(f"No images found in {os.path.abspath(self.image_dir)}. Check path and extensions.")
            return

        logger.info(f"Found {len(files)} image files. Grouping...")

        for f in files:
            subject_id, view_char = self._parse_filename(f)
            
            if not subject_id:
                continue

            if subject_id not in self.triplets:
                self.triplets[subject_id] = FaceTriplet(subject_id=subject_id)

            # Assign path to correct slot
            if view_char == 'f':
                self.triplets[subject_id].frontal_path = f
            elif view_char == 't':
                self.triplets[subject_id].three_quarter_path = f
            elif view_char == 'p':
                self.triplets[subject_id].profile_path = f

    def get_complete_subjects(self) -> List[FaceTriplet]:
        """Returns a list of FaceTriplet objects that have all 3 views."""
        complete = [t for t in self.triplets.values() if t.is_complete]
        logger.info(f"Processing complete. Found {len(complete)} valid subjects ready for 3D generation.")
        return complete

    def get_incomplete_subjects(self) -> List[FaceTriplet]:
        """Returns a list of subjects missing one or more views (useful for debugging)."""
        return [t for t in self.triplets.values() if not t.is_complete]

# --- Usage Example ---
if __name__ == "__main__":
    # Ensure the directory exists for testing
    test_dir = "images/bm"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created placeholder directory {test_dir}. Please add images like 'nbmf001.jpg' to test.")

    manager = ImageManager(image_dir=test_dir)
    manager.scan_directory()
    
    valid_subjects = manager.get_complete_subjects()
    
    if valid_subjects:
        print(f"\n--- Ready for Processing ({len(valid_subjects)}) ---")
        for subj in valid_subjects[:5]: # Show first 5
            print(f"ID: {subj.subject_id}")
            print(f"  Front: {os.path.basename(subj.frontal_path)}")
            print(f"  3/4:   {os.path.basename(subj.three_quarter_path)}")
            print(f"  Side:  {os.path.basename(subj.profile_path)}")
    else:
        print("\nNo complete triplets found.")
