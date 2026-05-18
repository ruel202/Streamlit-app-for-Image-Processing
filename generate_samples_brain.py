"""
generate_samples.py — Creates 3 synthetic grayscale sample images
that mimic CT texture for testing without real patient data.

Run once before launching the app:
    python generate_samples_brain.py
"""

import numpy as np
from PIL import Image
import os

os.makedirs("sample_images", exist_ok=True)

rng = np.random.default_rng(42)

# ── 1. Healthy brain ───────────────────────────────────────────────────
# White matter core (bright, smooth)
brain = rng.normal(loc=180, scale=8, size=(512, 512))

# Gray matter ring (slightly darker, more variable)
# Simulate cortical ribbon ~40px thick around WM
cy, cx = 256, 256
Y, X = np.ogrid[:512, :512]
r = np.sqrt((X - cx)**2 + (Y - cy)**2)
gm_mask = (r > 120) & (r < 160)
brain[gm_mask] = rng.normal(140, 15, brain[gm_mask].shape)

# CSF / ventricles (dark, very uniform)
# Two lateral ventricle ovals
for vx, vy in [(220, 256), (292, 256)]:
    vm = (((X - vx)/18)**2 + ((Y - vy)/35)**2) < 1
    brain[vm] = rng.normal(30, 4, brain[vm].shape)

# Skull ring (bright)
skull_mask = (r > 170) & (r < 190)
brain[skull_mask] = rng.normal(220, 6, brain[skull_mask].shape)

# Outside skull = air (black)
brain[r > 190] = rng.normal(5, 2, brain[r > 190].shape)

brain=np.clip(brain, 0, 255).astype(np.uint8)
Image.fromarray(brain).save("sample_images/brain_healthy.png")
print("Saved: sample_images/brain_healthy.png")

# ── 2.Brain_lesion:Same slice with a white matter lesion ───────────────────────────────────────────────────
# Replace np.ogrid with np.meshgrid — gives true 2D arrays
Y, X = np.meshgrid(np.arange(512), np.arange(512), indexing='ij')

lesion_img = brain.copy().astype(np.float64)

lmask = (((X - 200)/20)**2 + ((Y - 230)/15)**2) < 1

lesion_img[lmask] = rng.normal(210, 35, lesion_img[lmask].shape)
lesion_img[lmask] += 15 * np.sin(X[lmask] / 3.0) * np.cos(Y[lmask] / 3.0)

lesion_img = np.clip(lesion_img, 0, 255).astype(np.uint8)
Image.fromarray(lesion_img).save("sample_images/brain_lesion.png")
print("Saved: sample_images/brain_lesion.png")

# ── 3. Brain Glioma: Tumor with necrotic core ─────────────────────────────────────────────
# Large heterogeneous mass with ring enhancement pattern
tumor_mask = (((X - 280)/50)**2 + ((Y - 256)/45)**2) < 1
necrotic_core = (((X - 280)/25)**2 + ((Y - 256)/22)**2) < 1

brain_tumor = brain.copy()
brain_tumor[tumor_mask] = rng.normal(160, 40, brain_tumor[tumor_mask].shape)
brain_tumor[necrotic_core] = rng.normal(50, 10, brain_tumor[necrotic_core].shape)

brain_tumor=np.clip(brain_tumor, 0, 255).astype(np.uint8)
Image.fromarray(brain_tumor).save("sample_images/brain_glioma.png")
print("Saved: sample_images/brain_glioma.png")