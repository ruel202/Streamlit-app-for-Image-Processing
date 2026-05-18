"""
visualization.py — All matplotlib figures returned for st.pyplot().
Dark-themed, publication-style plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.feature import graycomatrix
from scipy.ndimage import gaussian_filter

# ── Shared style ─────────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GRID_COL  = "#30363d"
TEXT_COL  = "#e6edf3"
COLOR_A   = "#58a6ff"   # ROI A — blue
COLOR_B   = "#f78166"   # ROI B — coral

def _base_style():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    GRID_COL,
        "axes.labelcolor":   TEXT_COL,
        "axes.titlecolor":   TEXT_COL,
        "xtick.color":       TEXT_COL,
        "ytick.color":       TEXT_COL,
        "text.color":        TEXT_COL,
        "grid.color":        GRID_COL,
        "legend.facecolor":  PANEL_BG,
        "legend.edgecolor":  GRID_COL,
        "font.family":       "monospace",
    })


# ── Histogram ────────────────────────────────────────────────────────────────────

def plot_histogram(roi_a: np.ndarray, roi_b: np.ndarray):
    """Overlaid intensity histograms with KDE smoothing."""
    _base_style()
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=DARK_BG)

    for roi, color, label in [(roi_a, COLOR_A, "ROI A"), (roi_b, COLOR_B, "ROI B")]:
        counts, bins = np.histogram(roi.flatten(), bins=128, range=(0, 256))
        centers = (bins[:-1] + bins[1:]) / 2
        # Smooth for KDE-like appearance
        smooth = gaussian_filter(counts.astype(float), sigma=2)
        smooth = smooth / smooth.max()
        ax.fill_between(centers, smooth, alpha=0.25, color=color)
        ax.plot(centers, smooth, color=color, lw=1.5, label=label)

    ax.set_xlabel("Pixel Intensity (0–255)")
    ax.set_ylabel("Normalized Count")
    ax.set_title("Intensity Histogram — ROI A vs ROI B")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── GLCM Matrix Heatmap ──────────────────────────────────────────────────────────

def plot_glcm_matrix(roi: np.ndarray):
    """Heatmap of the GLCM at distance=1, angle=0°."""
    _base_style()
    levels = 64
    roi_q = (roi / 256.0 * levels).astype(np.uint8)
    roi_q = np.clip(roi_q, 0, levels - 1)

    glcm = graycomatrix(roi_q, distances=[1], angles=[0],
                        levels=levels, symmetric=True, normed=True)
    matrix = glcm[:, :, 0, 0]

    fig, ax = plt.subplots(figsize=(4, 4), facecolor=DARK_BG)
    im = ax.imshow(matrix, cmap="magma", aspect="auto",
                   norm=mcolors.PowerNorm(gamma=0.4))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("GLCM (d=1, θ=0°)")
    ax.set_xlabel("Gray level j")
    ax.set_ylabel("Gray level i")
    fig.tight_layout()
    return fig


# ── Polar Feature Chart ──────────────────────────────────────────────────────────

def plot_polar_features(glcm_a: dict, glcm_b: dict, angles_deg: list):
    """
    Radar/polar chart of GLCM contrast per angle for ROI A and ROI B.
    A perfect circle = isotropic texture.
    """
    _base_style()

    n = len(angles_deg)
    if n < 2:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "Need ≥2 angles for polar chart",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    angles_rad = [np.deg2rad(a) for a in angles_deg] + [np.deg2rad(angles_deg[0])]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"},
                           facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    for glcm, color, label in [(glcm_a, COLOR_A, "ROI A"), (glcm_b, COLOR_B, "ROI B")]:
        vals = glcm.get("_contrast_per_angle", [0] * n)
        vals_closed = list(vals) + [vals[0]]
        ax.plot(angles_rad, vals_closed, color=color, lw=2, label=label, marker="o")
        ax.fill(angles_rad, vals_closed, color=color, alpha=0.12)

    ax.set_thetagrids(angles_deg, labels=[f"{a}°" for a in angles_deg])
    ax.set_title("GLCM Contrast per Angle\n(circle = isotropic)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(color=GRID_COL)
    ax.tick_params(colors=TEXT_COL)
    fig.tight_layout()
    return fig


# ── FFT Power Spectrum ───────────────────────────────────────────────────────────

def plot_power_spectrum(freqs_a, power_a, freqs_b, power_b):
    """Log-log radial power spectrum comparison."""
    _base_style()
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)

    for freqs, power, color, label in [
        (freqs_a, power_a, COLOR_A, "ROI A"),
        (freqs_b, power_b, COLOR_B, "ROI B"),
    ]:
        # Skip DC component (index 0)
        f = freqs[1:]
        p = power[1:]
        p_smooth = gaussian_filter(np.log10(p + 1e-10), sigma=1.5)
        ax.plot(np.log10(f + 1e-10), p_smooth, color=color, lw=2, label=label)
        ax.fill_between(np.log10(f + 1e-10), p_smooth, alpha=0.15, color=color)

    ax.set_xlabel("log₁₀ Spatial Frequency (cycles/px)")
    ax.set_ylabel("log₁₀ Power")
    ax.set_title("Radial Power Spectrum — Frequency Domain Texture")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate slope regions
    ax.axvline(x=-1.3, color=GRID_COL, ls="--", lw=0.8, alpha=0.6)
    ax.text(-1.25, ax.get_ylim()[0] * 0.98 if ax.get_ylim()[0] < 0 else ax.get_ylim()[0] + 0.1,
            "← coarse  |  fine →", color=TEXT_COL, fontsize=8, alpha=0.7)

    fig.tight_layout()
    return fig


# ── Entropy Map ──────────────────────────────────────────────────────────────────

def plot_entropy_map(entropy_map: np.ndarray, original: np.ndarray):
    """
    Entropy heatmap overlaid (alpha blend) on the original CT slice.
    Also shows side-by-side: original | entropy | overlay.
    """
    _base_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor=DARK_BG)

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original (CLAHE)")
    axes[0].axis("off")

    im = axes[1].imshow(entropy_map, cmap="inferno")
    axes[1].set_title("Entropy Map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(original, cmap="gray", vmin=0, vmax=255)
    ent_norm = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-10)
    axes[2].imshow(ent_norm, cmap="inferno", alpha=0.45)
    axes[2].set_title("Overlay (entropy on image)")
    axes[2].axis("off")

    fig.suptitle("Sliding-Window Shannon Entropy", color=TEXT_COL, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig
