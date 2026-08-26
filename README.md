# Radiomics Texture Lab

## Goal 

| Level | Method | What it captures |
|---|---|---|
| 1 | First-order statistics | Intensity distribution (mean, variance, entropy, skewness…) |
| 2 | GLCM (second-order) | Spatial co-occurrence of pixel intensity pairs |
| 3 | FFT radial power spectrum | Texture scale and frequency-domain structure |

Plus a **sliding-window entropy map** for spatial heterogeneity visualization.

https://huggingface.co/ruel202/Radiomic_TextureLab

## Local Setup

```bash
git clone https://huggingface.co/spaces/ruel202/radiomics-texture-lab
cd radiomics-texture-lab

pip install -r requirements.txt
pip install geometric_kernels
python generate_samples_brain.py

streamlit run app.py
```

## Project Structure

```
app.py               # Streamlit UI — layout, controls, tab rendering
processing.py        # Image loading, CLAHE, ROI extraction, entropy map
metrics.py           # First-order stats, GLCM features, FFT power spectrum
visualization.py     # All matplotlib figures (dark-themed)
sample_images/       # 3 samples images based on the states of brain (HEALTHY, LESION, GLIOMA)
requirements.txt
README.md
docs/
  design_choices.md  # Technical rationale for every design decision
```

## Sample Images

Metrics collected from the public domains data sets
- `brain_healthy.png` — normal brain - ground truth T1 slice, is the simualtion which has the most similarity charac based on the tissue intensities  
- `brain_lesion.png` — region containing white matter (higher GLCM contrast)
- `brain_glioma.png` — tumor with necrotic core

## Known Limitations

- **2D only** — single image slices; no DICOM series or volumetric analysis
- **No DICOM reader** — export your DICOM to PNG/JPG before uploading
- **ROI must be ≥ 16×16 px** for GLCM to be meaningful
- **Entropy map is slow** for large images (>1024px); resize first
- **Radial power spectrum assumes isotropy** — strongly directional textures (ribs, vessels) are better characterized by the GLCM polar chart
- **GLCM quantized to 64 levels** — reduces sparsity in small ROIs but loses fine intensity detail
