# utils/convert_jp2.py
import sys
import os
import glob
import SimpleITK as sitk

# Ensure project root is in sys.path
sys.path.append(os.path.abspath("."))
from config import ORGAN_CONFIG

def convert_jp2_stack_to_nifti(jp2_dir, output_nii_path, spacing=None):
    """
    Converts a directory of 2D JP2 slice images into a single 3D NIfTI (.nii.gz) volume.
    """
    if os.path.exists(output_nii_path):
        print(f"[✓] Output NIfTI volume already exists: {output_nii_path} (Skipping conversion)")
        return True

    jp2_files = sorted(glob.glob(os.path.join(jp2_dir, "**", "*.jp2"), recursive=True))
    if not jp2_files:
        print(f"[WARNING] No JP2 files found in {jp2_dir}")
        return False
    
    print(f"[*] Found {len(jp2_files)} slices in {jp2_dir}. Converting to NIfTI volume...")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(jp2_files)
    image = reader.Execute()
    
    if spacing is not None:
        image.SetSpacing([float(s) for s in spacing])
    
    os.makedirs(os.path.dirname(output_nii_path), exist_ok=True)
    sitk.WriteImage(image, output_nii_path)
    print(f"[✓] Saved 3D volume to: {output_nii_path}")
    return True

if __name__ == "__main__":
    for organ, info in ORGAN_CONFIG.items():
        convert_jp2_stack_to_nifti(
            jp2_dir=info["raw_dir"],
            output_nii_path=info["path"],
            spacing=info["spacing"]
        )
