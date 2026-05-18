"""
metrics.py — Spatial statistics: first-order, GLCM (second-order), FFT power spectrum.
"""

import numpy as np
from scipy import stats as scipy_stats
from skimage.feature import graycomatrix, graycoprops


# ── First-order statistics ───────────────────────────────────────────────────────

def first_order_stats(roi: np.ndarray) -> dict:
    """
    Compute first-order intensity statistics over the ROI.
    These are computed independently per pixel — no spatial context.

    Returns a dict with:
        mean, std, variance, skewness, kurtosis, entropy,
        p10 (10th percentile), p90 (90th percentile), energy
    """
    flat = roi.flatten().astype(np.float64)

    # Shannon entropy from histogram
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    nonzero = hist[hist > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))

    return {
        "Mean":       float(np.mean(flat)),
        "Std Dev":    float(np.std(flat)),
        "Variance":   float(np.var(flat)),
        "Skewness":   float(scipy_stats.skew(flat)),
        "Kurtosis":   float(scipy_stats.kurtosis(flat)),
        "Entropy":    float(entropy),
        "10th %ile":  float(np.percentile(flat, 10)),
        "90th %ile":  float(np.percentile(flat, 90)),
        "Energy":     float(np.sum((flat / 255.0) ** 2) / len(flat)),
    }


# ── GLCM — Second-order spatial statistics ───────────────────────────────────────

def compute_glcm_features(
    roi: np.ndarray,
    distances: list,
    angles: list,
) -> dict:
    """
    Compute GLCM-based texture features.

    Design notes:
    - We quantize to 64 levels to keep the matrix tractable (256×256 GLCM
      is very sparse for small ROIs and slows rendering).
    - Isotropy score = 1 / (1 + std_across_angles). Higher = more isotropic.
    - Features are averaged across distances, returned per angle.

    Args:
        roi:       uint8 grayscale ROI
        distances: list of int pixel offsets, e.g. [1, 3, 5]
        angles:    list of float angles in radians, e.g. [0, π/4, π/2, 3π/4]

    Returns:
        dict with scalar summaries: contrast, homogeneity, energy,
        correlation, entropy, isotropy_score, plus per-angle arrays
        for polar plotting.
    """
    # Quantize to 64 gray levels
    levels = 64
    roi_q = (roi / 256.0 * levels).astype(np.uint8)
    roi_q = np.clip(roi_q, 0, levels - 1)

    glcm = graycomatrix(
        roi_q,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )
    # glcm shape: (levels, levels, len(distances), len(angles))

    def _prop(name):
        # Average over distances, keep per-angle
        return graycoprops(glcm, name).mean(axis=0)  # shape: (len(angles),)

    contrast    = _prop("contrast")
    homogeneity = _prop("homogeneity")
    energy      = _prop("energy")
    correlation = _prop("correlation")

    # GLCM entropy (not a skimage property — compute manually)
    glcm_mean = glcm.mean(axis=2)  # average over distances → (levels, levels, angles)
    eps = 1e-10
    glcm_ent = np.array([
        -np.sum(glcm_mean[:, :, a] * np.log2(glcm_mean[:, :, a] + eps))
        for a in range(len(angles))
    ])

    # Isotropy score: low std across angles = isotropic
    isotropy_score = float(1.0 / (1.0 + np.std(contrast)))

    return {
        # Scalar summaries (mean over angles)
        "contrast":       float(contrast.mean()),
        "homogeneity":    float(homogeneity.mean()),
        "energy":         float(energy.mean()),
        "correlation":    float(correlation.mean()),
        "entropy":        float(glcm_ent.mean()),
        "isotropy_score": isotropy_score,
        # Per-angle arrays (for polar chart)
        "_contrast_per_angle":    contrast.tolist(),
        "_homogeneity_per_angle": homogeneity.tolist(),
        "_energy_per_angle":      energy.tolist(),
        "_correlation_per_angle": correlation.tolist(),
        "_entropy_per_angle":     glcm_ent.tolist(),
        "_n_angles":              len(angles),
    }


# ── FFT Radial Power Spectrum ────────────────────────────────────────────────────

def compute_power_spectrum(roi: np.ndarray) -> tuple:
    """
    Compute the 1D radial power spectrum of the ROI via 2D FFT.

    Design notes:
    - Hanning window applied before FFT to suppress spectral leakage
      (boundary discontinuities cause artefact energy at all frequencies).
    - Radial averaging assumes texture isotropy — valid for parenchyma,
      less so for strongly oriented structures. Angular asymmetry is then
      better captured by GLCM polar chart.
    - Returns normalized spatial frequencies in cycles/pixel [0, 0.5].

    Returns:
        freqs: 1D array of spatial frequencies (cycles/pixel)
        power: 1D array of radially averaged power (log scale recommended)
    """
    roi_f = roi.astype(np.float64)

    # Hanning window (2D outer product)
    wy = np.hanning(roi_f.shape[0])
    wx = np.hanning(roi_f.shape[1])
    window_2d = np.outer(wy, wx)
    roi_windowed = roi_f * window_2d

    # 2D FFT, shift zero frequency to center
    fft2 = np.fft.fftshift(np.fft.fft2(roi_windowed))
    power_2d = np.abs(fft2) ** 2

    # Radial average
    h, w = power_2d.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices((h, w))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)

    max_r = min(cx, cy)
    radial_power = np.zeros(max_r)
    for radius in range(max_r):
        mask = r == radius
        if mask.sum() > 0:
            radial_power[radius] = power_2d[mask].mean()

    # Normalize frequency axis to cycles/pixel
    freqs = np.arange(max_r) / (2.0 * max_r)

    return freqs, radial_power
