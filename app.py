import base64
import io
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="SRoughnessLab Pro | Solomon Scientific",
    page_icon="SR LOGO.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# STYLE
# ==========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root {
    --navy:#0b1120; --navy-light:#1a2540; --gold:#c9a84c;
    --gold-light:#e2c97e; --gold-dim:#9c7a32; --white:#ffffff;
    --offwhite:#f8fafc; --text:#1e293b; --muted:#64748b;
    --border:#e2e8f0; --blue:#3a7bd5; --red:#e05252; --green:#3db87a;
}
html, body, [class*="css"] {font-family:'IBM Plex Sans',sans-serif;color:var(--text)}
.stApp {background:#fff}
[data-testid="stSidebar"] {background:#fff!important;border-right:1px solid var(--border)}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label {color:var(--text)!important}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {
 color:var(--gold-dim)!important;font-size:.75rem;letter-spacing:.15em;text-transform:uppercase
}
.stButton>button {background:linear-gradient(135deg,var(--gold-dim),var(--gold))!important;
 color:var(--navy)!important;border:0!important;border-radius:4px!important;font-weight:600!important}
[data-testid="stDownloadButton"]>button {background:var(--offwhite)!important;color:var(--navy)!important;
 border:1px solid var(--border)!important}
[data-testid="stDataFrame"] {border:1px solid var(--border)!important;border-radius:6px!important}
[data-testid="stTabs"] [role="tab"] {font-size:.78rem!important;font-weight:600!important;text-transform:uppercase!important}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# UI HELPERS
# ==========================================================
def get_base64_of_bin_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render_header():
    logo = "SR LOGO.png"
    if os.path.exists(logo):
        icon = (
            f'<img src="data:image/png;base64,{get_base64_of_bin_file(logo)}" '
            'style="width:54px;height:54px;border-radius:8px;object-fit:contain;background:white">'
        )
    else:
        icon = '<div style="font-size:2rem">🔬</div>'
    st.markdown(
        f"""
<div style="background:linear-gradient(135deg,#0b1120,#1a2540);padding:1.4rem 2rem;
 border-radius:8px;border:1px solid rgba(201,168,76,.35);display:flex;align-items:center;
 gap:1.4rem;margin:1rem 0 1.4rem">
 {icon}
 <div><div style="font-family:'Playfair Display';font-size:1.75rem;font-weight:700;color:#f8fafc">
 SRoughnessLab <span style="color:#c9a84c">Pro</span></div>
 <div style="font-size:.72rem;color:#a8b4c8;letter-spacing:.18em;text-transform:uppercase">
 Surface Metrology Analysis Suite · Solomon Scientific · © 2026</div></div>
</div>
""",
        unsafe_allow_html=True,
    )


def section_title(text, icon=""):
    st.markdown(
        f"""
<div style="background:linear-gradient(90deg,#0b1120,#1a2540);padding:.6rem 1.1rem;
 border-radius:6px;border-left:4px solid #c9a84c;margin:1.3rem 0 .9rem;color:#f8fafc;
 font-size:.8rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase">{icon} {text}</div>
""",
        unsafe_allow_html=True,
    )


def metric_card(label, value, unit=""):
    return f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-top:3px solid #c9a84c;
 border-radius:6px;padding:1rem 1.2rem;box-shadow:0 2px 8px rgba(0,0,0,.03)">
 <div style="font-size:.68rem;color:#64748b;letter-spacing:.1em;text-transform:uppercase;font-weight:600">{label}</div>
 <div style="font-family:'IBM Plex Mono';font-size:1.3rem;font-weight:600;color:#1e293b">
 {value}<span style="font-size:.7rem;color:#64748b;margin-left:4px">{unit}</span></div>
</div>
"""


def natural_sort_key(value):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(value))]


# ==========================================================
# ROUGHNESS AND WENZEL CALCULATIONS
# ==========================================================
def compute_roughness_params(signal):
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]
    if signal.size == 0:
        return (np.nan,) * 6
    ra = np.mean(np.abs(signal))
    rq = np.sqrt(np.mean(signal**2))
    rt = np.max(signal) - np.min(signal)
    rsk = stats.skew(signal, bias=False) if signal.size > 2 else np.nan
    rku = stats.kurtosis(signal, fisher=True, bias=False) if signal.size > 3 else np.nan
    sorted_s = np.sort(signal)
    rz = np.mean(sorted_s[-5:] - sorted_s[:5]) if signal.size >= 10 else np.nan
    return ra, rq, rz, rt, rsk, rku


def compute_profile_roughness_ratio(length_mm, roughness_um):
    """Return the 2D profile-length ratio r_L and RMS slope Rdq.

    x is expected in mm and z in micrometres. The x coordinates are converted
    to micrometres before Euclidean segment lengths are calculated.
    """
    x_mm = np.asarray(length_mm, dtype=float)
    z_um = np.asarray(roughness_um, dtype=float)
    valid = np.isfinite(x_mm) & np.isfinite(z_um)
    x_um = x_mm[valid] * 1000.0
    z_um = z_um[valid]
    if x_um.size < 2:
        return np.nan, np.nan

    dx_um = np.diff(x_um)
    dz_um = np.diff(z_um)
    keep = dx_um > 0
    dx_um = dx_um[keep]
    dz_um = dz_um[keep]
    if dx_um.size == 0:
        return np.nan, np.nan

    projected = np.sum(dx_um)
    actual = np.sum(np.hypot(dx_um, dz_um))
    if projected <= 0:
        return np.nan, np.nan

    r_l = actual / projected
    rdq = np.sqrt(np.mean((dz_um / dx_um) ** 2))
    return float(r_l), float(rdq)


def wenzel_corrected_angle(apparent_angle_deg, roughness_ratio):
    """theta_Y = acos(cos(theta_W)/r). Angles are supplied and returned in degrees."""
    theta_w = float(apparent_angle_deg)
    r = float(roughness_ratio)
    if not np.isfinite(theta_w) or not np.isfinite(r) or r < 1.0:
        return np.nan
    argument = np.clip(np.cos(np.deg2rad(theta_w)) / r, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(argument)))


def iso_sigma(lambda_c_mm, dx_mm):
    if not np.isfinite(dx_mm) or dx_mm <= 0:
        dx_mm = 0.001
    return (lambda_c_mm / (2.0 * np.pi)) / dx_mm


def export_excel(df, sheet_name):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        for i, col in enumerate(df.columns):
            lengths = df[col].astype(str).map(len) if not df.empty else pd.Series([0])
            width = min(max(len(str(col)), int(lengths.max())) + 2, 40)
            ws.set_column(i, i, width)
    return out.getvalue()


# ==========================================================
# DATA LOADER
# ==========================================================
class RoughnessLoader:
    def __init__(self):
        self.targets = {"Ra": ["ra"], "Rq": ["rq"], "Rz": ["rz"], "Rt": ["rt"]}

    @staticmethod
    def clean_value(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(value).replace(",", ".").strip())
        return float(match.group()) if match else np.nan

    def process_files(self, uploaded_files, sample_id, filter_type, sg_window, lambda_c):
        summaries, profiles = [], {}
        for uploaded in uploaded_files:
            try:
                xl = pd.ExcelFile(uploaded)
                summary = {"Sample": sample_id, "File": uploaded.name, "Filter": filter_type}

                for sheet in xl.sheet_names:
                    scan = xl.parse(sheet, header=None)
                    for row in range(min(len(scan), 100)):
                        for col in range(len(scan.columns)):
                            text = str(scan.iloc[row, col]).lower().strip()
                            for key, keywords in self.targets.items():
                                if key in summary or not any(k in text for k in keywords):
                                    continue
                                value = np.nan
                                if col + 1 < len(scan.columns):
                                    value = self.clean_value(scan.iloc[row, col + 1])
                                if np.isnan(value) and row + 1 < len(scan):
                                    value = self.clean_value(scan.iloc[row + 1, col])
                                if np.isfinite(value):
                                    summary[key] = value

                data_sheet = next((s for s in xl.sheet_names if "DATA" in s.upper()), xl.sheet_names[-1])
                uploaded.seek(0)
                profile = pd.read_excel(uploaded, sheet_name=data_sheet, usecols=[4, 5], header=None)
                profile.columns = ["Length_mm", "Amplitude_um"]
                for col in profile.columns:
                    profile[col] = pd.to_numeric(
                        profile[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
                    )
                profile = profile.dropna().reset_index(drop=True)
                if profile.empty:
                    st.error(f"{uploaded.name}: no numeric profile data found in columns E and F.")
                    continue

                x = profile["Length_mm"].to_numpy()
                z = profile["Amplitude_um"].to_numpy()
                if filter_type == "ISO Gaussian (λc)":
                    dx = float(np.mean(np.diff(x))) if len(x) > 1 else 0.001
                    sigma = max(iso_sigma(lambda_c, dx), 0.01)
                    profile["Form"] = gaussian_filter1d(z, sigma=sigma)
                    profile["Roughness"] = z - profile["Form"]
                elif filter_type == "Savitzky-Golay":
                    max_odd = len(profile) if len(profile) % 2 == 1 else len(profile) - 1
                    window = min(sg_window, max_odd)
                    window = max(window, 5)
                    if window % 2 == 0:
                        window -= 1
                    profile["Form"] = savgol_filter(z, window_length=window, polyorder=3)
                    profile["Roughness"] = z - profile["Form"]
                else:
                    profile["Form"] = np.mean(z)
                    profile["Roughness"] = z - np.mean(z)

                ra, rq, rz, rt, rsk, rku = compute_roughness_params(profile["Roughness"])
                r_l, rdq = compute_profile_roughness_ratio(profile["Length_mm"], profile["Roughness"])
                summary.update(
                    {
                        "Ra_calc": ra,
                        "Rq_calc": rq,
                        "Rz_calc": rz,
                        "Rt_calc": rt,
                        "Rsk": rsk,
                        "Rku": rku,
                        "r_L": r_l,
                        "Rdq": rdq,
                    }
                )
                profile["Sample"] = sample_id
                profiles[uploaded.name] = profile
                summaries.append(summary)
            except Exception as exc:
                st.error(f"Error processing {uploaded.name}: {exc}")
        return pd.DataFrame(summaries), profiles


# ==========================================================
# SESSION STATE
# ==========================================================
for key, default in {
    "master_df": pd.DataFrame(),
    "profile_dict": {},
    "legend_map": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown("### 1 · Data Input")
    with st.form("input_form", clear_on_submit=True):
        sample_name = st.text_input("Sample ID", "Sample A")
        files = st.file_uploader("Upload replicate files", type=["xlsx"], accept_multiple_files=True)
        filter_name = st.selectbox("Detrending filter", ["ISO Gaussian (λc)", "Savitzky-Golay", "None"])
        lambda_c, sg_window = 0.8, 51
        if filter_name == "ISO Gaussian (λc)":
            lambda_c = st.number_input("Cutoff wavelength λc (mm)", 0.1, value=0.8, step=0.1)
        elif filter_name == "Savitzky-Golay":
            sg_window = st.slider("S-G window", 5, 151, 51, step=2)
        submit = st.form_submit_button("Add Sample Batch", use_container_width=True)

    if submit and files:
        loader = RoughnessLoader()
        new_summary, new_profiles = loader.process_files(
            files, sample_name, filter_name, sg_window, lambda_c
        )
        st.session_state.master_df = pd.concat(
            [st.session_state.master_df, new_summary], ignore_index=True
        )
        st.session_state.profile_dict.update(new_profiles)
        st.success(f"Added {len(new_summary)} valid file(s).")

    if not st.session_state.master_df.empty:
        st.markdown("### 2 · Manage Data")
        samples = sorted(st.session_state.master_df["Sample"].unique(), key=natural_sort_key)
        delete_sample = st.selectbox("Delete batch", ["Select"] + samples)
        if st.button("Delete Batch", use_container_width=True) and delete_sample != "Select":
            removed = st.session_state.master_df.loc[
                st.session_state.master_df["Sample"] == delete_sample, "File"
            ].tolist()
            st.session_state.master_df = st.session_state.master_df.loc[
                st.session_state.master_df["Sample"] != delete_sample
            ].reset_index(drop=True)
            for name in removed:
                st.session_state.profile_dict.pop(name, None)
            st.rerun()

    if st.button("Reset Entire Study", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ==========================================================
# MAIN
# ==========================================================
render_header()
df = st.session_state.master_df
profiles = st.session_state.profile_dict

if df.empty:
    st.info("Upload one or more profilometer Excel files from the sidebar to begin.")
    st.stop()

ra_source = "Ra" if "Ra" in df.columns else "Ra_calc"
mean_ra = df[ra_source].mean() if ra_source in df else np.nan
std_ra = df[ra_source].std() if ra_source in df else np.nan
k1, k2, k3, k4 = st.columns(4)
k1.markdown(metric_card("Batches", df["Sample"].nunique()), unsafe_allow_html=True)
k2.markdown(metric_card("Replicates", len(df), "files"), unsafe_allow_html=True)
k3.markdown(metric_card("Mean Ra", f"{mean_ra:.3f}", "µm"), unsafe_allow_html=True)
k4.markdown(metric_card("Mean rL", f"{df['r_L'].mean():.6f}"), unsafe_allow_html=True)

TAB_LABELS = [
    "📋 Dataset",
    "📉 Roughness Trends",
    "🎨 Profiles",
    "📈 PSD",
    "💧 Wenzel Correction",
    "💾 Export",
    "📖 Methods",
]
tabs = st.tabs(TAB_LABELS)

with tabs[0]:
    section_title("Dataset", "📋")
    st.dataframe(df, use_container_width=True, height=430)

with tabs[1]:
    section_title("Inter-Sample Trends", "📉")
    parameters = [c for c in ["Ra", "Rq", "Rz", "Rt", "Ra_calc", "Rq_calc", "r_L", "Rdq"] if c in df]
    parameter = st.selectbox("Parameter", parameters)
    summary = df.groupby("Sample", as_index=False)[parameter].agg(["mean", "std", "count"]).reset_index()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summary["Sample"], y=summary["mean"], mode="lines+markers",
            error_y=dict(type="data", array=summary["std"], visible=True),
            line=dict(color="#0b1120", width=2.5), marker=dict(color="#c9a84c", size=10),
        )
    )
    fig.update_layout(
        template="plotly_white", width=850, height=520,
        xaxis_title="Sample ID", yaxis_title=parameter,
        font=dict(family="IBM Plex Sans"),
    )
    st.plotly_chart(fig, use_container_width=False)

with tabs[2]:
    section_title("Filtered Roughness Profiles", "🎨")
    selected_file = st.selectbox("Profile file", sorted(profiles, key=natural_sort_key), key="profile")
    p = profiles[selected_file]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p["Length_mm"], y=p["Roughness"], mode="lines", line=dict(color="#3a7bd5")))
    fig.update_layout(template="plotly_white", width=900, height=520,
                      xaxis_title="Profile length (mm)", yaxis_title="Roughness amplitude (µm)")
    st.plotly_chart(fig, use_container_width=False)

with tabs[3]:
    section_title("Power Spectral Density", "📈")
    psd_file = st.selectbox("Profile file", sorted(profiles, key=natural_sort_key), key="psd")
    p = profiles[psd_file]
    signal = p["Roughness"].to_numpy()
    x = p["Length_mm"].to_numpy()
    dx = np.mean(np.diff(x))
    freq = np.fft.rfftfreq(len(signal), d=dx)
    power = np.abs(np.fft.rfft(signal)) ** 2
    mask = freq > 0
    fig = go.Figure(go.Scatter(x=freq[mask], y=power[mask], mode="lines", line=dict(color="#9b59b6")))
    fig.update_layout(template="plotly_white", width=900, height=520,
                      xaxis_type="log", yaxis_type="log",
                      xaxis_title="Spatial frequency (cycles/mm)", yaxis_title="Power density")
    st.plotly_chart(fig, use_container_width=False)

with tabs[4]:
    section_title("Profile-Based Wenzel Correction", "💧")
    st.latex(r"\cos\theta_W=r_L\cos\theta_Y")
    st.latex(r"\theta_Y=\cos^{-1}\left(\frac{\cos\theta_W}{r_L}\right)")

    valid = df.loc[df["r_L"].notna() & (df["r_L"] >= 1)].copy()
    if valid.empty:
        st.warning("No valid rL values are available.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            w_file = st.selectbox("Surface profile file", sorted(valid["File"], key=natural_sort_key))
        with c2:
            theta_w = st.number_input(
                "Measured apparent contact angle θW (°)", 0.0, 180.0, 90.0, 0.1, format="%.2f"
            )

        selected = valid.loc[valid["File"] == w_file].iloc[0]
        r_l = float(selected["r_L"])
        rdq = float(selected["Rdq"])
        theta_y = wenzel_corrected_angle(theta_w, r_l)
        change = theta_y - theta_w

        a, b, c, d = st.columns(4)
        a.markdown(metric_card("Sample", selected["Sample"]), unsafe_allow_html=True)
        b.markdown(metric_card("Profile Ratio rL", f"{r_l:.6f}"), unsafe_allow_html=True)
        c.markdown(metric_card("RMS Slope Rdq", f"{rdq:.6f}"), unsafe_allow_html=True)
        d.markdown(metric_card("Corrected θY", f"{theta_y:.2f}", "°"), unsafe_allow_html=True)

        result = pd.DataFrame(
            {
                "Sample": [selected["Sample"]],
                "Profile_File": [w_file],
                "Apparent_CA_deg": [theta_w],
                "Profile_Roughness_Ratio_rL": [r_l],
                "RMS_Profile_Slope_Rdq": [rdq],
                "Corrected_Young_CA_deg": [theta_y],
                "Correction_deg": [change],
            }
        )
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Wenzel Result",
            export_excel(result, "Wenzel_Correction"),
            "Wenzel_Contact_Angle_Correction.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.warning(
            "rL is a two-dimensional profile-length approximation. The formal Wenzel factor is a "
            "three-dimensional actual-to-projected area ratio. Apply each rL only to the surface "
            "represented by the same profile file."
        )

with tabs[5]:
    section_title("Export", "💾")
    st.download_button(
        "Download Summary",
        export_excel(df, "Summary"),
        "SRoughnessLab_Summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if profiles:
        blocks = []
        for file_name in sorted(profiles, key=natural_sort_key):
            block = profiles[file_name][["Length_mm", "Amplitude_um", "Roughness"]].copy()
            stem = os.path.splitext(file_name)[0]
            block.columns = [f"{stem}_X_mm", f"{stem}_Amplitude_um", f"{stem}_Roughness_um"]
            blocks.append(block.reset_index(drop=True))
        profile_export = pd.concat(blocks, axis=1)
        st.download_button(
            "Download All Profiles",
            export_excel(profile_export, "Profiles"),
            "SRoughnessLab_All_Profiles.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

with tabs[6]:
    section_title("Methods", "📖")
    st.markdown(
        r"""
### Profile roughness ratio

The application estimates a two-dimensional profile-length roughness ratio:

\[
r_L=\frac{\sum_{i=1}^{n-1}\sqrt{(\Delta x_i)^2+(\Delta z_i)^2}}
{\sum_{i=1}^{n-1}\Delta x_i}
\]

The horizontal coordinates are converted from millimetres to micrometres before calculating
segment lengths. The RMS profile slope is:

\[
R_{dq}=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n-1}
\left(\frac{\Delta z_i}{\Delta x_i}\right)^2}
\]

### Wenzel correction

\[
\theta_Y=\cos^{-1}\left(\frac{\cos\theta_W}{r_L}\right)
\]

The result should be reported as a profile-based approximation because the formal Wenzel
roughness factor is a three-dimensional actual-to-projected area ratio. The Wenzel model also
assumes complete liquid penetration into the surface texture.
"""
    )
