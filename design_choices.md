# Design Choices

Technical rationale for every non-obvious decision in this app.

---

## 1. Why CLAHE before GLCM?

Raw CT/X-ray images often have poor local contrast due to the wide dynamic range of tissue densities. CLAHE (Contrast Limited Adaptive Histogram Equalization) normalizes local contrast without globally clipping the histogram — this is critical because GLCM is sensitive to the full intensity range. Without normalization, the co-occurrence matrix would be dominated by the narrow band of intensities where most pixels cluster, making features like contrast artificially low and homogeneity artificially high.

**Trade-off:** CLAHE introduces local contrast enhancement that can amplify noise in uniform regions. The `clipLimit=2.0` default is conservative — increase it for very low-contrast images, decrease for noisy ones.

---

## 2. Why 64 gray levels for GLCM?

A full 256×256 GLCM is extremely sparse for small ROIs (common in medical image analysis). Sparsity means most features are driven by a handful of co-occurring pairs, making them unreliable. Quantizing to 64 levels:
- Keeps the matrix dense enough for stable feature estimates
- Reduces computation time for interactive use
- Matches standard radiomics practice (pyradiomics default: 32–64 levels)

**Trade-off:** Fine intensity distinctions (e.g., subtle Hounsfield Unit differences) are lost.

---

## 3. Why a Hanning window before FFT?

Without windowing, the FFT treats the ROI as if it tiles periodically — but the left and right edges of a CT patch rarely match, creating artificial step discontinuities. These inject energy at all frequencies (spectral leakage), making the power spectrum unreliable. The Hanning window tapers the edges smoothly to zero, concentrating spectral energy where it actually belongs.

**Trade-off:** Windowing reduces effective spatial resolution (you effectively analyze a slightly smaller region). For very small ROIs (<32px), this loss is significant.

---

## 4. Why radial averaging for the power spectrum?

The 2D FFT produces a 2D power map. Radial averaging collapses it to 1D by averaging power at each distance from the DC component. This assumes the texture is **statistically isotropic** — i.e., looks similar in all directions. For lung parenchyma and liver, this is approximately true.

For strongly anisotropic structures (ribs, vessels), radial averaging hides directional information — which is precisely why the GLCM polar chart exists as a complementary view.

---

## 5. Why isotropy score = 1 / (1 + std_across_angles)?

The isotropy score needs to be:
- High when angular variation is low (isotropic)
- Low when angular variation is high (anisotropic)
- Bounded in [0, 1] for easy interpretation

`1 / (1 + σ)` satisfies all three. It's a standard formulation used in directional statistics. The `+1` prevents division by zero and keeps the denominator always ≥ 1.

---

## 6. Sliding window size vs. entropy map resolution

Larger windows → smoother entropy map, captures larger-scale heterogeneity, less sensitive to single-pixel noise.
Smaller windows → noisier map, captures fine-grained texture variation, more sensitive to local anomalies.

For a 512×512 CT slice:
- Window 7–11: fine detail, noisy baseline
- Window 15–21: good balance for lesion detection context
- Window 25–31: coarse, good for identifying large structural differences

The window size slider in the sidebar lets users explore this trade-off directly.

---

## 7. Why not use pyradiomics directly?

`pyradiomics` is the clinical-grade radiomics library and implements the full IBSI (Image Biomarker Standardisation Initiative) feature set. It is not used here because:
1. It requires SimpleITK and a full DICOM/NIfTI pipeline — too heavyweight for a teaching app
2. It is a black box from a pedagogical perspective — the goal here is to show the math
3. Installation on Hugging Face Spaces free tier is unreliable

The app implements the same core features (first-order, GLCM) from scratch using `scikit-image` and `scipy`, which makes the computation transparent and auditable.
