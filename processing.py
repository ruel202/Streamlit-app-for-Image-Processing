"""
processing.py — Image loading, preprocessing, and ROI extraction.
"""
import nibabel as nib
import numpy as np
import cv2
from PIL import Image
import io


def load_image(source) -> np.ndarray:
    if isinstance(source, str):
        img = Image.open(source).convert("L")
    else:
        # Streamlit UploadedFile
        raw_bytes = source.read()
        img = Image.open(io.BytesIO(raw_bytes)).convert("L")

    return np.array(img, dtype=np.uint8)


def load_nifti_slice(file, slice_index=None) -> np.ndarray:
    img = nib.load(file)
    vol = img.get_fdata()  # shape: (H, W, num_slices)
    
    if slice_index is None:
        slice_index = vol.shape[2] // 2  # middle slice
    
    slc = vol[:, :, slice_index]
    # Normalize to uint8
    slc = (slc - slc.min()) / (slc.max() - slc.min()) * 255
    return slc.astype(np.uint8)

def apply_clahe(img: np.ndarray, clip_limit: float = 1.5, tile_grid: tuple = (8, 8)) -> np.ndarray:
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def extract_roi(img: np.ndarray, bbox: tuple) -> np.ndarray:
    x1, y1, x2, y2 = bbox

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box: x2 must be > x1 and y2 must be > y1. Got {bbox}")

    h, w = img.shape
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        raise ValueError(f"ROI is empty after clipping to image bounds. bbox={bbox}, image shape={img.shape}")

    return roi


def sliding_window_entropy(img: np.ndarray, window: int = 15) -> np.ndarray:
   
    from skimage.filters.rank import entropy as rank_entropy
    from skimage.morphology import disk

    # Use a square approximation via disk with appropriate radius
    # skimage rank.entropy uses a structuring element
    radius = window // 2
    selem = _square_selem(window)

    # rank.entropy expects uint8
    ent_map = rank_entropy(img, selem)
    return ent_map.astype(np.float32)


def _square_selem(size: int) -> np.ndarray:
    """Return a square boolean structuring element of given side length."""
    return np.ones((size, size), dtype=np.uint8)
