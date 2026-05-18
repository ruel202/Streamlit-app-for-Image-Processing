"""
generate_samples.py — Creates 3 synthetic grayscale sample images
that mimic CT texture for testing without real patient data.

Run once before launching the app:
    python generate_samples.py
"""

import numpy as np
from PIL import Image
import os

os.makedirs("sample_images", exist_ok=True)

rng = np.random.default_rng(42)

# ── 1. Healthy lung parenchyma ───────────────────────────────────────────────────
# Fine-grained, relatively uniform low-intensity texture
lung = rng.normal(loc=60, scale=12, size=(512, 512))
# Add subtle vascular-like structures (thin bright lines)
for _ in range(8):
    x = rng.integers(50, 462)
    y = rng.integers(50, 462)
    length = rng.integers(30, 80)
    angle = rng.uniform(0, np.pi)
    for t in np.linspace(0, length, 200):
        xi = int(x + t * np.cos(angle))
        yi = int(y + t * np.sin(angle))
        if 0 <= xi < 512 and 0 <= yi < 512:
            lung[yi, xi] = 160 + rng.normal(0, 10)

lung = np.clip(lung, 0, 255).astype(np.uint8)
Image.fromarray(lung).save("sample_images/lung_healthy.png")
print("Saved: sample_images/lung_healthy.png")

# ── 2. Lung with nodule region ───────────────────────────────────────────────────
# Same parenchyma + a heterogeneous circular nodule
nodule_img = lung.copy().astype(np.float64)
cy, cx = 256, 256
radius = 45
y_idx, x_idx = np.ogrid[:512, :512]
mask = (x_idx - cx)**2 + (y_idx - cy)**2 <= radius**2

# Nodule: higher mean intensity, more variance, rough edges
nodule_texture = rng.normal(loc=160, scale=30, size=(512, 512))
nodule_texture += 20 * np.sin(x_idx / 4.0) * np.cos(y_idx / 4.0)
nodule_img[mask] = nodule_texture[mask]

nodule_img = np.clip(nodule_img, 0, 255).astype(np.uint8)
Image.fromarray(nodule_img).save("sample_images/lung_nodule.png")
print("Saved: sample_images/lung_nodule.png")

# ── 3. Liver with density variation ─────────────────────────────────────────────
# Smooth spatial gradient + layered texture variation
x_grid = np.linspace(0, 1, 512)
y_grid = np.linspace(0, 1, 512)
XX, YY = np.meshgrid(x_grid, y_grid)

liver = 100 + 40 * XX + 20 * np.sin(5 * np.pi * YY) + rng.normal(0, 8, (512, 512))
# Add a low-density region (cyst-like)
liver_mask = (x_grid[np.newaxis, :] - 0.7)**2 + (y_grid[:, np.newaxis] - 0.35)**2 < 0.015
liver[liver_mask] = rng.normal(40, 5, liver[liver_mask].shape)

liver = np.clip(liver, 0, 255).astype(np.uint8)
Image.fromarray(liver).save("sample_images/liver_density.png")
print("Saved: sample_images/liver_density.png")

print("\nAll sample images generated in sample_images/")
print("Now run: streamlit run app.py")
