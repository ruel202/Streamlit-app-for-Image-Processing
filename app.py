import streamlit as st
import numpy as np
from PIL import Image
import os

from processing import load_image,load_nifti_slice, apply_clahe, extract_roi, sliding_window_entropy
from metrics import first_order_stats, compute_glcm_features, compute_power_spectrum
from visualization import (
    plot_histogram,
    plot_glcm_matrix,
    plot_polar_features,
    plot_power_spectrum,
    plot_entropy_map,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Radiomics Texture Lab",
    page_icon="🧠 Radiomics Texture Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0d1117; }
  .block-container { padding-top: 1.5rem; }
  h1 { letter-spacing: -1px; }
  .stTabs [data-baseweb="tab"] { font-size: 0.85rem; font-weight: 600; }
  .metric-box {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(" Radiomics Texture Lab")
    st.caption("Multi-scale spatial statistics for medical images")
    st.divider()

    st.subheader("📂 Image Input")
    uploaded = st.file_uploader("Upload image (NifTi/PNG/JPG/DCM-exported)", type=["nii","png", "jpg", "jpeg"])

    
    st.markdown("**Or use a sample:**")
    sample_dir = "sample_images"
    samples = sorted([f for f in os.listdir(sample_dir) if f.endswith((".png", ".jpg"))]) \
        if os.path.isdir(sample_dir) else []

   
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None

    cols = st.columns(len(samples)) if samples else []
    for i, s in enumerate(samples):
        with cols[i]:
            if st.button(s.replace(".png","").replace(".jpg",""), use_container_width=True):
               st.session_state.selected_sample = os.path.join(sample_dir, s)
               st.session_state.clicks = []   # reset clicks on sample switch

    selected_sample = st.session_state.selected_sample

    st.divider()
    st.subheader("⚙️ GLCM Parameters")
    glcm_distances = st.multiselect(
        "Distances (px)", options=[1, 2, 3, 5, 8, 10],
        default=[1, 3], help="Spatial offset for co-occurrence matrix"
    )
    glcm_angles_deg = st.multiselect(
        "Angles (°)", options=[0, 45, 90, 135],
        default=[0, 45, 90, 135]
    )
    entropy_window = st.slider("Sliding window size (px)", 5, 31, 15, step=2)

# Clear selected sample if user uploads a file
if uploaded:
    st.session_state.selected_sample = None
    selected_sample = None

    st.divider()
    st.subheader("🧩 ROI Settings")
    st.info("Draw ROI A and ROI B directly on the image panels below.")

# ── Load image ──────────────────────────────────────────────────────────────────
raw_img = None
n_slices = None

if uploaded:
    if uploaded.name.endswith(".nii") or uploaded.name.endswith(".nii.gz"):
        import tempfile, os, nibabel as nib

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".nii.gz" if uploaded.name.endswith(".gz") else ".nii"
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        vol_shape = nib.load(tmp_path).shape
        n_slices  = vol_shape[2]

        slice_index = st.sidebar.slider(
            "🧠 NIfTI slice",
            min_value=0,
            max_value=n_slices - 1,
            value=n_slices // 2,
            help=f"Volume has {n_slices} axial slices"
        )

        raw_img = load_nifti_slice(tmp_path, slice_index)
        os.unlink(tmp_path)

        # Reset clicks when slice changes
        if st.session_state.get("last_slice") != slice_index:
            st.session_state.clicks = []
            st.session_state.last_slice = slice_index

    else:
        raw_img = load_image(uploaded)

elif selected_sample:
    raw_img = load_image(selected_sample)

# Reset clicks when image source changes
current_source = uploaded.name if uploaded else (selected_sample or "none")
if st.session_state.get("last_source") != current_source:
    st.session_state.clicks = []
    st.session_state.last_source = current_source

if raw_img is None:
    st.markdown("## 👈 Upload an image or pick a sample to begin")
    st.markdown("""
    This app teaches **three levels of spatial texture analysis** on medical images:
 
    | Level | Method | What it captures |
    |---|---|---|
    | 1 | First-order statistics | Intensity distribution (mean, variance, entropy…) |
    | 2 | GLCM (second-order) | Spatial relationships between pixel pairs |
    | 3 | FFT power spectrum | Texture scale & frequency-domain structure |
 
    Select two ROIs to compare healthy vs. abnormal tissue.
    """)
    st.stop()
    

# ── Preprocessing ────────────────────────────────────────────────────────────────
clahe_img = apply_clahe(raw_img)

st.markdown("## Image Overview")
col_orig, col_clahe = st.columns(2)
with col_orig:
    st.markdown("**Original (grayscale)**")
    st.image(raw_img, use_container_width=True, clamp=True)
with col_clahe:
    st.markdown("**CLAHE-enhanced**")
    st.image(clahe_img, use_container_width=True, clamp=True)

# ── ROI Selection ────────────────────────────────────────────────────────────────
from streamlit_image_coordinates import streamlit_image_coordinates
import PIL.ImageDraw as ImageDraw

st.divider()
st.markdown("## ROI Selection")
st.caption("Click on the image — A1 → A2 → B1 → B2.")

if "clicks" not in st.session_state:
    st.session_state.clicks = []

# Scale factors
IMG_DISPLAY_WIDTH = min(raw_img.shape[1], 700)
scale_x = raw_img.shape[1] / IMG_DISPLAY_WIDTH
scale_y = raw_img.shape[0] / IMG_DISPLAY_WIDTH

def to_display(pt):
    return (int(pt[0] / scale_x), int(pt[1] / scale_y))

# Draw markers on display-sized image
display_img = Image.fromarray(raw_img).convert("RGB")
draw_img    = display_img.copy()
draw        = ImageDraw.Draw(draw_img)

colors = ["#58a6ff", "#58a6ff", "#f78166", "#f78166"]
labels = ["A1", "A2", "B1", "B2"]

for i, pt in enumerate(st.session_state.clicks):
    dx, dy = to_display(pt)
    r = 6
    draw.ellipse([dx-r, dy-r, dx+r, dy+r], fill=colors[i], outline="white", width=2)
    draw.text((dx+9, dy-9), labels[i], fill=colors[i])

# NEW — sorts to guarantee top-left / bottom-right order
def to_display_rect(p1, p2):
    """Convert two original-pixel points to a sorted display-pixel rectangle."""
    d1 = to_display(p1)
    d2 = to_display(p2)
    return [
        (min(d1[0], d2[0]), min(d1[1], d2[1])),  # top-left
        (max(d1[0], d2[0]), max(d1[1], d2[1])),  # bottom-right
    ]

if len(st.session_state.clicks) >= 2:
    draw.rectangle(
        to_display_rect(st.session_state.clicks[0], st.session_state.clicks[1]),
        outline="#58a6ff", width=2
    )

if len(st.session_state.clicks) >= 4:
    draw.rectangle(
        to_display_rect(st.session_state.clicks[2], st.session_state.clicks[3]),
        outline="#f78166", width=2
    )

st.markdown("**Click on the image — up to 4 points (A1 → A2 → B1 → B2):**")
coords = streamlit_image_coordinates(draw_img, key="main_image", width=IMG_DISPLAY_WIDTH)

if coords is not None and len(st.session_state.clicks) < 4:
    new_pt = (int(coords["x"] * scale_x), int(coords["y"] * scale_y))
    if len(st.session_state.clicks) == 0 or st.session_state.clicks[-1] != new_pt:
        st.session_state.clicks.append(new_pt)
        st.rerun()

col_status, col_reset = st.columns([3, 1])
with col_status:
    n = len(st.session_state.clicks)
    messages = [
        "👆 Click top-left corner of **ROI A**",
        "👆 Click bottom-right corner of **ROI A**",
        "👆 Click top-left corner of **ROI B**",
        "👆 Click bottom-right corner of **ROI B**",
        "✅ Both ROIs defined — scroll down to see analysis",
    ]
    st.info(messages[n])

with col_reset:
    if st.button("🔄 Reset clicks", use_container_width=True):
        st.session_state.clicks = []
        st.rerun()

if len(st.session_state.clicks) < 4:
    st.stop()

def ordered_bbox(p1, p2):
    return (min(p1[0], p2[0]), min(p1[1], p2[1]),
            max(p1[0], p2[0]), max(p1[1], p2[1]))

bbox_a = ordered_bbox(st.session_state.clicks[0], st.session_state.clicks[1])
bbox_b = ordered_bbox(st.session_state.clicks[2], st.session_state.clicks[3])

ca, cb = st.columns(2)
with ca:
    st.markdown(f"**ROI A** — `{bbox_a[0]},{bbox_a[1]}` → `{bbox_a[2]},{bbox_a[3]}` "
                f"({bbox_a[2]-bbox_a[0]}×{bbox_a[3]-bbox_a[1]} px)")
with cb:
    st.markdown(f"**ROI B** — `{bbox_b[0]},{bbox_b[1]}` → `{bbox_b[2]},{bbox_b[3]}` "
                f"({bbox_b[2]-bbox_b[0]}×{bbox_b[3]-bbox_b[1]} px)")
    

# bbox_a and bbox_b are already in original image pixel coords
# because we applied scale_x / scale_y when registering clicks

try:
    roi_a = extract_roi(clahe_img, bbox_a)
    roi_b = extract_roi(clahe_img, bbox_b)
except ValueError as e:
    st.error(f"ROI error: {e}")
    st.stop()

# Guard: minimum size for GLCM to be meaningful
if roi_a.shape[0] < 4 or roi_a.shape[1] < 4:
    st.warning("ROI A is too small — click further apart.")
    st.stop()

if roi_b.shape[0] < 4 or roi_b.shape[1] < 4:
    st.warning("ROI B is too small — click further apart.")
    st.stop()

# Preview
col_ra, col_rb = st.columns(2)
with col_ra:
    st.markdown("**ROI A preview**")
    st.image(roi_a, use_container_width=True, clamp=True)
with col_rb:
    st.markdown("**ROI B preview**")
    st.image(roi_b, use_container_width=True, clamp=True)
# ── Analysis Tabs ───────────────────────────────────────────────────────────────
st.divider()
tab1, tab2, tab3, tab4, tab5= st.tabs([
    "📊 First-Order Statistics",
    "🔲 GLCM (Second-Order)",
    "🌊 FFT Power Spectrum",
    "🌡️ Entropy Map",
    "🫀 GP on Tumor Surface",
])

# ── Tab 1: First-order ──────────────────────────────────────────────────────────
with tab1:
    st.markdown("### First-Order Intensity Statistics")
    st.caption("WM should show high mean, low variance. "
           "Lesions show elevated variance and entropy.")
    stats_a = first_order_stats(roi_a)
    stats_b = first_order_stats(roi_b)
    st.markdown("#### Feature Values")
    col_a, col_b = st.columns(2)
    for col, stats, label in [(col_a, stats_a, "ROI A"), (col_b, stats_b, "ROI B")]:
        with col:
            st.markdown(f"**{label}**")
            for k, v in stats.items():
                st.markdown(
                    f'<div class="metric-box"><b>{k}</b><br/>'
                    f'<span style="font-size:1.4rem">{v:.4f}</span></div>',
                    unsafe_allow_html=True
                )

    st.markdown("#### Intensity Histogram")
    st.pyplot(plot_histogram(roi_a, roi_b))
    st.caption("**Interpretation:** A wider histogram indicates higher intensity variance. "
               "Higher entropy signals more complex, heterogeneous tissue.")
    

    

    


# ── Tab 2: GLCM ─────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### GLCM — Gray-Level Co-occurrence Matrix")
    st.caption("WM/GM boundary produces high GLCM contrast. "
           "CSF shows extreme homogeneity — nearly all pixel pairs are similar intensity.")

    if not glcm_distances or not glcm_angles_deg:
        st.warning("Select at least one distance and one angle in the sidebar.")
    else:
        angles_rad = [np.deg2rad(a) for a in glcm_angles_deg]

        glcm_a = compute_glcm_features(roi_a, glcm_distances, angles_rad)
        glcm_b = compute_glcm_features(roi_b, glcm_distances, angles_rad)

        st.markdown("#### GLCM Matrix Heatmap (distance=1, angle=0°)")
        col_ga, col_gb = st.columns(2)
        with col_ga:
            st.markdown("**ROI A**")
            st.pyplot(plot_glcm_matrix(roi_a))
        with col_gb:
            st.markdown("**ROI B**")
            st.pyplot(plot_glcm_matrix(roi_b))

        st.markdown("#### Feature Comparison Table")
        features = ["contrast", "homogeneity", "energy", "correlation", "entropy", "isotropy_score"]
        rows = []
        for f in features:
            rows.append({
                "Feature": f,
                "ROI A": f"{glcm_a.get(f, 0):.4f}",
                "ROI B": f"{glcm_b.get(f, 0):.4f}",
            })
        st.table(rows)

        st.markdown("#### Angular Feature Polar Chart")
        st.pyplot(plot_polar_features(glcm_a, glcm_b, glcm_angles_deg))
        st.caption("**Interpretation:** A circular polar plot indicates isotropic texture (uniform in all directions). "
                   "Asymmetry reveals directional structure (e.g., vessels, ribs). "
                   "The isotropy score is the inverse of angular standard deviation.")

# ── Tab 3: FFT Power Spectrum ────────────────────────────────────────────────────
with tab3:
    st.markdown("### Radial Power Spectrum (2D FFT)")
    st.caption("Cortical folding creates quasi-periodic structures visible as "
           "a spectral bump around 0.05–0.1 cycles/px.")

    freqs_a, power_a = compute_power_spectrum(roi_a)
    freqs_b, power_b = compute_power_spectrum(roi_b)

    st.pyplot(plot_power_spectrum(freqs_a, power_a, freqs_b, power_b))
    st.caption("**Interpretation:** Steep slope = smooth/homogeneous texture. "
               "Flat slope = heterogeneous, complex tissue. "
               "Bumps at specific frequencies suggest periodic structures (e.g., trabecular bone pattern).")

# ── Tab 4: Entropy Map ───────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Sliding-Window Entropy Map")
    st.caption("Lesions appear as bright islands in the entropy map. "
           "Ventricles appear dark (maximally homogeneous).")
    with st.spinner("Computing entropy map (may take a few seconds)..."):
        entropy_map = sliding_window_entropy(clahe_img, window=entropy_window)

    st.pyplot(plot_entropy_map(entropy_map, clahe_img))
    st.caption("**Interpretation:** Uniform tissue (healthy brain) shows low entropy (dark). "
               "Lesions, tumors, or complex boundaries appear bright — spatially heterogeneous. "
               "Use the window size slider to control spatial resolution vs. sensitivity trade-off.")

# ── Tab 5: GP on Tumor Surface ────────────────────────────────────────────────
with tab5:
    st.markdown("### 🧬 Gaussian Process Regression on Tumor Surface Mesh")
    st.caption(
        "Upload a 3D tumor mesh (.obj). A synthetic radiomic signal is sampled from a GP prior, "
        "then reconstructed from sparse 'biopsy' observations using a geometry-aware kernel. "
        "Compare SPDE / Geodesic / RBF kernels and visualise posterior mean + uncertainty."
    )
 
    # ── Lazy imports ──────────────────────────────────────────────────────────
    try:
        import potpourri3d as pp3d
        import plotly.graph_objects as go
        import plotly.express as px
        from scipy.special import kv, gamma as scipy_gamma
        from scipy import stats as sp_stats
        _mesh_libs_ok = True
    except ImportError as _e:
        st.error(f"Missing dependency: {_e}. Run `pip install potpourri3d plotly`.")
        _mesh_libs_ok = False
 
    if _mesh_libs_ok:
 
        # ── Mesh upload ───────────────────────────────────────────────────────
        mesh_file = st.file_uploader(
            "Upload tumor mesh (.obj)", type=["obj"], key="mesh_upload"
        )
        st.markdown("**No mesh yet?** Download a free example:")
        st.code("https://github.com/alecjacobson/common-3d-test-models  (e.g. spot.obj, armadillo.obj)")
        st.caption(
            "For a real tumor mesh: export an .obj from 3D Slicer after segmenting "
            "a lesion in a NIfTI scan."
        )
 
        if mesh_file is not None:
            import tempfile
 
            @st.cache_data(show_spinner="Loading mesh…")
            def _load_mesh(data: bytes):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".obj") as tmp:
                    tmp.write(data)
                return pp3d.read_mesh(tmp.name)
 
            v_mesh, f_mesh = _load_mesh(mesh_file.read())
            n_verts = len(v_mesh)
            st.success(f"Mesh loaded — {n_verts:,} vertices, {len(f_mesh):,} faces")
 
            # ── Controls ──────────────────────────────────────────────────────
            col_ctrl1, col_ctrl2 = st.columns(2)
            with col_ctrl1:
                gp_kernel = st.selectbox(
                    "Prediction kernel", ["SPDE", "Geodesic", "RBF"],
                    help="Kernel the GP uses to interpolate the radiomic signal."
                )
                gt_kernel = st.selectbox(
                    "Ground-truth kernel (signal generation)", ["SPDE", "Geodesic", "RBF"],
                    help="Kernel used to *sample* the synthetic ground-truth field."
                )
            with col_ctrl2:
                gp_nu    = st.slider("Smoothness ν", 0.5, 3.0, 1.5, 0.5,  key="gp_nu")
                gp_kappa = st.slider("Lengthscale κ", 0.01, 0.30, 0.07, 0.01, key="gp_kappa")
                gp_noise = st.slider("Observation noise σ", 0.01, 0.30, 0.05, 0.01, key="gp_noise")
                gp_ntrain = st.slider(
                    "Biopsy / training points", 5, min(300, n_verts - 1), 50, key="gp_ntrain"
                )
 
            # ── Kernel builders ───────────────────────────────────────────────
            def _matern_val(d, nu, kappa):
                if np.isclose(d, 0):
                    return 1.0
                x = np.sqrt(2 * nu) * d / kappa
                return (2 ** (1 - nu) / scipy_gamma(nu)) * (x ** nu) * kv(nu, x)
 
            @st.cache_data(show_spinner="Computing geodesic kernel…")
            def _K_geo(v_b, f_b, idx1_t, idx2_t, nu, kappa):
                v_arr = np.array(v_b); f_arr = np.array(f_b)
                idx1 = list(idx1_t);   idx2  = list(idx2_t)
                solver = pp3d.MeshHeatMethodDistanceSolver(v_arr, f_arr)
                K = np.zeros((len(idx1), len(idx2)))
                for i, vi in enumerate(idx1):
                    dists = solver.compute_distance(vi)
                    for j, vj in enumerate(idx2):
                        K[i, j] = _matern_val(dists[vj], nu, kappa)
                return K
 
            def _K_rbf(v_b, idx1_t, idx2_t, kappa):
                v_arr = np.array(v_b)
                v1 = v_arr[list(idx1_t)]; v2 = v_arr[list(idx2_t)]
                diff = v1[:, None, :] - v2[None, :, :]
                return np.exp(-np.sum(diff ** 2, axis=2) / (2 * kappa ** 2))
            
 
            def _K_spde(v_b, f_b, idx1_t, idx2_t, nu, kappa):
                try:
                    from geometric_kernels.spaces  import Mesh as GKMesh
                    from geometric_kernels.kernels import MaternGeometricKernel
                    v_arr = np.array(v_b); f_arr = np.array(f_b)
                    space  = GKMesh(v_arr, f_arr)
                    kernel = MaternGeometricKernel(space)
                    params = kernel.init_params()
                    params["nu"]          = np.array([nu])
                    params["lengthscale"] = np.array([kappa])
                    X1 = np.array(list(idx1_t))[:, None]
                    X2 = np.array(list(idx2_t))[:, None]
                    return kernel.K(params, X1, X2)
                except ImportError:
                    st.warning("geometric_kernels not installed — falling back to Geodesic kernel.")
                    return _K_geo(v_b, f_b, idx1_t, idx2_t, nu, kappa)
 
            def _build_K(name, v_b, f_b, idx1_t, idx2_t, nu, kappa):
                if name == "Geodesic":
                    return _K_geo(v_b, f_b, idx1_t, idx2_t, nu, kappa)
                elif name == "SPDE":
                    return _K_spde(v_b, f_b, idx1_t, idx2_t, nu, kappa)
                elif name == "RBF":
                    return _K_rbf(v_b, idx1_t, idx2_t, kappa)
                else:
                    raise ValueError(name)
 
            # ── GP posterior (memory-safe, no n×n matrix) ─────────────────────
            def _gp_posterior(K_oo, K_oa, y_train, noise):
                """
                K_oo : (n_train, n_train)
                K_oa : (n_train, n_all)
                Returns mu (n_all,) and var (n_all,).
                k(x,x)=1 for normalised kernels so K_aa diagonal = ones.
                """
                n_tr = K_oo.shape[0]
                K_oo_n = K_oo + noise * np.eye(n_tr) + 1e-8 * np.eye(n_tr)
                L     = np.linalg.cholesky(K_oo_n)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                mu    = K_oa.T @ alpha                      # (n_all,)
                V     = np.linalg.solve(L, K_oa)           # (n_train, n_all)
                var   = np.clip(1.0 - np.sum(V ** 2, axis=0), 1e-8, None)  # (n_all,)
                return mu, var
 
            def _nlpd(mu, var, y_true):
                return 0.5 * float(np.mean(np.log(2 * np.pi * var) + (y_true - mu) ** 2 / var))
 
            # ── Mesh plot helper ──────────────────────────────────────────────
            def _mesh_fig(values, title, train_i=None, colorscale="Viridis"):
                fig = go.Figure(data=[go.Mesh3d(
                    x=v_mesh[:, 0], y=v_mesh[:, 1], z=v_mesh[:, 2],
                    i=f_mesh[:, 0], j=f_mesh[:, 1], k=f_mesh[:, 2],
                    intensity=values, colorscale=colorscale, opacity=1.0,
                    lighting=dict(ambient=0.5, diffuse=1.0, specular=0.4),
                    showscale=True,
                )])
                if train_i is not None:
                    fig.add_trace(go.Scatter3d(
                        x=v_mesh[list(train_i), 0],
                        y=v_mesh[list(train_i), 1],
                        z=v_mesh[list(train_i), 2],
                        mode="markers",
                        marker=dict(size=3, color="cyan", opacity=0.85),
                        name="Observed (biopsy)"
                    ))
                fig.update_layout(
                    title=dict(text=title, font=dict(size=13)),
                    scene=dict(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        zaxis=dict(visible=False),
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                    height=400,
                )
                return fig
 
            # ── Run button ────────────────────────────────────────────────────
            if st.button("▶ Run GP on Mesh", type="primary"):
 
                rng  = np.random.default_rng(42)
                perm = rng.permutation(n_verts)
 
                train_idx = tuple(perm[:gp_ntrain].tolist())
                test_idx  = tuple(perm[gp_ntrain:].tolist())
                all_idx   = tuple(range(n_verts))
 
                # hashable copies for cached functions
                v_t = tuple(map(tuple, v_mesh))
                f_t = tuple(map(tuple, f_mesh))
 
                # ── Ground truth: sample on a small subset, propagate ─────────
                # Never build n×n — use at most 300 anchor points.
                n_gt = min(300, n_verts)
                gt_anchor_idx = tuple(rng.permutation(n_verts)[:n_gt].tolist())
 
                with st.spinner("Sampling ground-truth radiomic field…"):
                    K_gt_oo = _build_K(gt_kernel, v_t, f_t,
                                       gt_anchor_idx, gt_anchor_idx, gp_nu, gp_kappa)
                    K_gt_oo += 1e-6 * np.eye(n_gt)
                    f_gt_anchors = rng.multivariate_normal(np.zeros(n_gt), K_gt_oo)
 
                    # Propagate anchor samples to full mesh via GP posterior mean
                    K_gt_oa = _build_K(gt_kernel, v_t, f_t,
                                       gt_anchor_idx, all_idx, gp_nu, gp_kappa)
                    alpha_gt  = np.linalg.solve(K_gt_oo, f_gt_anchors)
                    f_true_mesh = K_gt_oa.T @ alpha_gt          # (n_verts,)  — no n×n ever
 
                # ── Noisy train observations ──────────────────────────────────
                y_train_mesh = (
                    f_true_mesh[list(train_idx)]
                    + gp_noise * rng.standard_normal(gp_ntrain)
                )
 
                # ── Prediction kernel matrices ────────────────────────────────
                # K_oo : (n_train, n_train)   small
                # K_oa : (n_train, n_verts)   moderate — e.g. 50 × 111 k = 44 MB
                # K_aa : NOT built — posterior variance uses 1 - ||V||² trick
                with st.spinner(f"Computing {gp_kernel} kernel matrices…"):
                    K_oo = _build_K(gp_kernel, v_t, f_t,
                                    train_idx, train_idx, gp_nu, gp_kappa)
                    K_oa = _build_K(gp_kernel, v_t, f_t,
                                    train_idx, all_idx, gp_nu, gp_kappa)
 
                # ── Posterior ─────────────────────────────────────────────────
                with st.spinner("Computing GP posterior…"):
                    mu_all, var_all = _gp_posterior(K_oo, K_oa, y_train_mesh, gp_noise)
 
                # ── Metrics ───────────────────────────────────────────────────
                mu_test  = mu_all[list(test_idx)]
                var_test = var_all[list(test_idx)]
                y_test   = f_true_mesh[list(test_idx)]
 
                rmse_val = float(np.sqrt(np.mean((mu_test - y_test) ** 2)))
                nlpd_val = _nlpd(mu_test, var_test, y_test)
                coverage = float(np.mean(np.abs(y_test - mu_test) < 2 * np.sqrt(var_test)))
                error_log = np.log1p((mu_all - f_true_mesh) ** 2)
 
                # Next suggested biopsy: highest posterior variance outside train set
                cand_var = var_all.copy()
                cand_var[list(train_idx)] = -1.0
                next_biopsy_idx = int(np.argmax(cand_var))
 
                # ── Metrics display ───────────────────────────────────────────
                st.markdown("#### Results")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("RMSE (test)",    f"{rmse_val:.4f}")
                mc2.metric("NLPD (test)",    f"{nlpd_val:.4f}")
                mc3.metric("95% Coverage",   f"{coverage*100:.1f}%",
                           delta=f"{(coverage - 0.95)*100:+.1f} pp vs ideal")
                mc4.metric("Biopsy points",  f"{gp_ntrain} / {n_verts:,}")
 
                # ── 4-panel mesh plots ────────────────────────────────────────
                st.markdown("#### Surface Maps")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.plotly_chart(
                        _mesh_fig(f_true_mesh, "Ground-Truth Radiomic Field", train_idx),
                        use_container_width=True
                    )
                    st.plotly_chart(
                        _mesh_fig(error_log, "Log Squared Error", colorscale="Reds"),
                        use_container_width=True
                    )
                with col_m2:
                    st.plotly_chart(
                        _mesh_fig(mu_all, f"GP Posterior Mean ({gp_kernel})", train_idx),
                        use_container_width=True
                    )
                    var_norm = (var_all - var_all.min()) / (var_all.max() - var_all.min() + 1e-12)
                    st.plotly_chart(
                        _mesh_fig(var_norm, "Posterior Uncertainty (normalised)",
                                  colorscale="Hot"),
                        use_container_width=True
                    )
 
                # ── Active learning: next biopsy ──────────────────────────────
                st.markdown("#### 🎯 Suggested Next Biopsy Location")
                st.caption(
                    "The surface point with the highest posterior variance — "
                    "optimal next sampling location under a maximum-uncertainty strategy."
                )
                nb_coord = v_mesh[next_biopsy_idx]
                st.info(
                    f"**Vertex #{next_biopsy_idx}** | "
                    f"coords ({nb_coord[0]:.3f}, {nb_coord[1]:.3f}, {nb_coord[2]:.3f}) | "
                    f"uncertainty = {var_all[next_biopsy_idx]:.4f}"
                )
                fig_next = _mesh_fig(var_all, "Next Biopsy Suggestion (red ◆)",
                                     train_i=train_idx, colorscale="Hot")
                fig_next.add_trace(go.Scatter3d(
                    x=[nb_coord[0]], y=[nb_coord[1]], z=[nb_coord[2]],
                    mode="markers",
                    marker=dict(size=8, color="red", symbol="diamond"),
                    name="Suggested next biopsy"
                ))
                st.plotly_chart(fig_next, use_container_width=True)
 
                # ── Variance vs geodesic distance ─────────────────────────────
                st.markdown("#### Uncertainty vs. Geodesic Distance to Nearest Biopsy Point")
                st.caption(
                    "A well-specified kernel produces a smooth monotone curve — "
                    "uncertainty grows as you move away from observations."
                )
                with st.spinner("Computing geodesic distances…"):
                    solver_dist = pp3d.MeshHeatMethodDistanceSolver(v_mesh, f_mesh)
                    # use at most 30 train points to keep this fast
                    sample_train = list(train_idx)[:30]
                    dist_stack   = np.stack(
                        [solver_dist.compute_distance(int(i)) for i in sample_train], axis=1
                    )
                    dist_to_train = dist_stack.min(axis=1)   # (n_verts,)
 
                scatter_fig = go.Figure()
                scatter_fig.add_trace(go.Scatter(
                    x=dist_to_train[list(test_idx)],
                    y=var_all[list(test_idx)],
                    mode="markers",
                    marker=dict(size=3, opacity=0.35, color="#58a6ff"),
                    name="Test vertices",
                ))
                scatter_fig.update_layout(
                    xaxis_title="Geodesic distance to nearest biopsy point",
                    yaxis_title="Posterior variance",
                    height=350,
                )
                st.plotly_chart(scatter_fig, use_container_width=True)
 
                # ── Calibration curve ─────────────────────────────────────────
                st.markdown("#### Calibration Curve")
                st.caption(
                    "Points on the diagonal = perfectly calibrated uncertainty. "
                    "Above = overconfident. Below = underconfident."
                )
                levels     = np.linspace(0.05, 0.99, 30)
                cov_actual = []
                for p in levels:
                    z = sp_stats.norm.ppf((1 + p) / 2)
                    cov_actual.append(
                        float(np.mean(np.abs(y_test - mu_test) < z * np.sqrt(var_test)))
                    )
 
                cal_fig = go.Figure()
                cal_fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    line=dict(dash="dash", color="gray"),
                    name="Perfect calibration",
                ))
                cal_fig.add_trace(go.Scatter(
                    x=levels.tolist(), y=cov_actual,
                    mode="lines+markers",
                    marker=dict(size=4),
                    line=dict(color="#f78166"),
                    name="Actual coverage",
                ))
                cal_fig.update_layout(
                    xaxis_title="Expected coverage level",
                    yaxis_title="Actual coverage",
                    height=350,
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                )
                st.plotly_chart(cal_fig, use_container_width=True)
 
                # ── Train–train kernel matrix ─────────────────────────────────
                st.markdown("#### Train–Train Kernel Matrix")
                st.caption("Reveals the similarity structure between observed biopsy points.")
                km_fig = px.imshow(
                    K_oo,
                    color_continuous_scale="Viridis",
                    title=f"{gp_kernel} kernel matrix (train × train)",
                )
                st.plotly_chart(km_fig, use_container_width=True)
 
        else:
            st.info("Upload a .obj mesh file above to enable GP-on-surface analysis.")
 
        # ── Data sources ──────────────────────────────────────────────────────
        with st.expander("📦 Where to find tumor mesh data"):
            st.markdown("""
**Synthetic / test meshes (no registration needed)**
- [common-3d-test-models](https://github.com/alecjacobson/common-3d-test-models) — spot.obj, armadillo.obj, bunny.obj
- [Thingi10K](https://ten-thousand-models.appspot.com/) — 10 000 real-world meshes
 
**Real tumor meshes from medical imaging**
1. **TCIA** — `https://www.cancerimagingarchive.net` — download CT/MRI DICOM series (e.g. TCGA-GBM).
2. Open in **3D Slicer** (free) → `Segment Editor` → paint the lesion → `Export to file` → `.obj`.
3. Upload the exported `.obj` here.
 
**Pre-segmented surface datasets**
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/) — liver, brain, prostate (NIfTI → convert with 3D Slicer)
- [SegTHOR](https://competitions.codalab.org/competitions/21145) — thoracic organ surfaces
 
**NIfTI → mesh (Python)**
```python
import nibabel as nib, pymcubes
seg = nib.load("tumor_seg.nii.gz").get_fdata()
verts, faces = pymcubes.marching_cubes(seg, 0.5)
pymcubes.export_obj(verts, faces, "tumor.obj")
```
""")
 