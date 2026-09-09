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

st.set_page_config(
    page_title="SRoughnessLab Pro | Solomon Scientific",
    page_icon="SR LOGO.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root {--navy:#0b1120;--navy-light:#1a2540;--gold:#c9a84c;--gold-dim:#9c7a32;--offwhite:#f8fafc;--text:#1e293b;--muted:#64748b;--border:#e2e8f0}
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;color:var(--text)}
.stApp{background:#fff}[data-testid="stSidebar"]{background:#fff!important;border-right:1px solid var(--border)}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:var(--gold-dim)!important;font-size:.75rem;letter-spacing:.15em;text-transform:uppercase}
.stButton>button{background:linear-gradient(135deg,var(--gold-dim),var(--gold))!important;color:var(--navy)!important;border:0!important;border-radius:4px!important;font-weight:600!important}
[data-testid="stDownloadButton"]>button{background:var(--offwhite)!important;color:var(--navy)!important;border:1px solid var(--border)!important}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:6px!important}
</style>
""", unsafe_allow_html=True)


def natural_sort_key(value):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(value))]


def get_logo_html():
    path = "SR LOGO.png"
    if not os.path.exists(path):
        return '<div style="font-size:2rem">🔬</div>'
    with open(path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    return f'<img src="data:image/png;base64,{encoded}" style="width:54px;height:54px;border-radius:8px;object-fit:contain;background:white">'


def render_header():
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0b1120,#1a2540);padding:1.4rem 2rem;border-radius:8px;border:1px solid rgba(201,168,76,.35);display:flex;align-items:center;gap:1.4rem;margin:1rem 0 1.4rem">
      {get_logo_html()}<div><div style="font-family:'Playfair Display';font-size:1.75rem;font-weight:700;color:#f8fafc">SRoughnessLab <span style="color:#c9a84c">Pro</span></div>
      <div style="font-size:.72rem;color:#a8b4c8;letter-spacing:.18em;text-transform:uppercase">Surface Metrology Analysis Suite · Solomon Scientific · © 2026</div></div>
    </div>""", unsafe_allow_html=True)


def section_title(text, icon=""):
    st.markdown(f'<div style="background:linear-gradient(90deg,#0b1120,#1a2540);padding:.6rem 1.1rem;border-radius:6px;border-left:4px solid #c9a84c;margin:1.3rem 0 .9rem;color:#f8fafc;font-size:.8rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase">{icon} {text}</div>', unsafe_allow_html=True)


def metric_card(label, value, unit=""):
    return f'<div style="background:#fff;border:1px solid #e2e8f0;border-top:3px solid #c9a84c;border-radius:6px;padding:1rem 1.2rem"><div style="font-size:.68rem;color:#64748b;text-transform:uppercase;font-weight:600">{label}</div><div style="font-family:IBM Plex Mono;font-size:1.3rem;font-weight:600">{value}<span style="font-size:.7rem;color:#64748b;margin-left:4px">{unit}</span></div></div>'


def compute_roughness_params(signal):
    z = np.asarray(signal, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return (np.nan,) * 6
    ra = np.mean(np.abs(z))
    rq = np.sqrt(np.mean(z ** 2))
    rt = np.ptp(z)
    rsk = stats.skew(z, bias=False) if z.size > 2 else np.nan
    rku = stats.kurtosis(z, fisher=True, bias=False) if z.size > 3 else np.nan
    ordered = np.sort(z)
    rz = np.mean(ordered[-5:] - ordered[:5]) if z.size >= 10 else np.nan
    return ra, rq, rz, rt, rsk, rku


def compute_profile_roughness_ratio(length_mm, roughness_um):
    x = np.asarray(length_mm, dtype=float)
    z = np.asarray(roughness_um, dtype=float)
    good = np.isfinite(x) & np.isfinite(z)
    x_um, z_um = x[good] * 1000.0, z[good]
    if x_um.size < 2:
        return np.nan, np.nan
    dx, dz = np.diff(x_um), np.diff(z_um)
    positive = dx > 0
    dx, dz = dx[positive], dz[positive]
    if dx.size == 0 or np.sum(dx) <= 0:
        return np.nan, np.nan
    r_l = np.sum(np.hypot(dx, dz)) / np.sum(dx)
    rdq = np.sqrt(np.mean((dz / dx) ** 2))
    return float(r_l), float(rdq)


def wenzel_corrected_angle(apparent_angle_deg, roughness_ratio):
    theta_w, r_l = float(apparent_angle_deg), float(roughness_ratio)
    if not np.isfinite(theta_w) or not np.isfinite(r_l) or r_l < 1:
        return np.nan
    value = np.clip(np.cos(np.deg2rad(theta_w)) / r_l, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(value)))


def iso_sigma(lambda_c_mm, dx_mm):
    return (lambda_c_mm / (2 * np.pi)) / dx_mm if np.isfinite(dx_mm) and dx_mm > 0 else 1.0


def standardize_roughness_columns(dataframe):
    """Guarantee Ra, Rq, Rz, and Rt without overwriting valid instrument values."""
    result = dataframe.copy()
    for reported, calculated in {"Ra":"Ra_calc", "Rq":"Rq_calc", "Rz":"Rz_calc", "Rt":"Rt_calc"}.items():
        if reported not in result.columns:
            result[reported] = np.nan
        result[reported] = pd.to_numeric(result[reported], errors="coerce")
        if calculated in result.columns:
            result[reported] = result[reported].combine_first(pd.to_numeric(result[calculated], errors="coerce"))
    for column in ["r_L", "Rdq"]:
        if column not in result.columns:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def export_excel(dataframe, sheet_name):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        dataframe.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        for i, column in enumerate(dataframe.columns):
            max_data = dataframe[column].astype(str).map(len).max() if not dataframe.empty else 0
            worksheet.set_column(i, i, min(max(len(str(column)), int(max_data)) + 2, 40))
    return output.getvalue()


class RoughnessLoader:
    PARAMETERS = ("Ra", "Rq", "Rz", "Rt")

    @staticmethod
    def clean_value(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.number)):
            return float(value)
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value).replace(",", "."))
        return float(match.group()) if match else np.nan

    @staticmethod
    def extract_parameter_from_text(value, parameter):
        if pd.isna(value):
            return np.nan
        pattern = rf"\b{re.escape(parameter)}\b\s*[:=]?\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        match = re.search(pattern, str(value).replace(",", "."), flags=re.IGNORECASE)
        return float(match.group(1)) if match else np.nan

    def extract_reported_parameters(self, excel_file):
        values = {}
        for sheet in excel_file.sheet_names:
            scan = excel_file.parse(sheet, header=None)
            for row in range(min(len(scan), 100)):
                for col in range(len(scan.columns)):
                    cell = scan.iloc[row, col]
                    for parameter in self.PARAMETERS:
                        if parameter in values or not re.search(rf"\b{parameter}\b", str(cell), re.I):
                            continue
                        value = self.extract_parameter_from_text(cell, parameter)
                        if not np.isfinite(value) and col + 1 < len(scan.columns):
                            value = self.clean_value(scan.iloc[row, col + 1])
                        if not np.isfinite(value) and row + 1 < len(scan):
                            value = self.clean_value(scan.iloc[row + 1, col])
                        if np.isfinite(value):
                            values[parameter] = value
        return values

    def process_files(self, uploaded_files, sample_id, filter_type, sg_window, lambda_c):
        summaries, profiles = [], {}
        for uploaded in uploaded_files:
            try:
                uploaded.seek(0)
                excel = pd.ExcelFile(uploaded)
                summary = {"Sample": sample_id, "File": uploaded.name, "Filter": filter_type}
                summary.update(self.extract_reported_parameters(excel))
                data_sheet = next((s for s in excel.sheet_names if "DATA" in s.upper()), excel.sheet_names[-1])
                uploaded.seek(0)
                profile = pd.read_excel(uploaded, sheet_name=data_sheet, usecols=[4, 5], header=None)
                profile.columns = ["Length_mm", "Amplitude_um"]
                for column in profile.columns:
                    profile[column] = pd.to_numeric(profile[column].astype(str).str.replace(",", ".", regex=False), errors="coerce")
                profile = profile.dropna().sort_values("Length_mm").drop_duplicates("Length_mm").reset_index(drop=True)
                if len(profile) < 5:
                    st.error(f"{uploaded.name}: fewer than five numeric x-z points were found in columns E and F.")
                    continue

                x, z = profile["Length_mm"].to_numpy(), profile["Amplitude_um"].to_numpy()
                if filter_type == "ISO Gaussian (λc)":
                    dx = float(np.median(np.diff(x)))
                    sigma = max(iso_sigma(lambda_c, dx), 0.01)
                    profile["Form"] = gaussian_filter1d(z, sigma=sigma, mode="nearest")
                elif filter_type == "Savitzky-Golay":
                    maximum = len(profile) if len(profile) % 2 else len(profile) - 1
                    window = min(int(sg_window), maximum)
                    window = max(window, 5)
                    if window % 2 == 0:
                        window -= 1
                    profile["Form"] = savgol_filter(z, window_length=window, polyorder=min(3, window - 1))
                else:
                    profile["Form"] = np.mean(z)
                profile["Roughness"] = z - profile["Form"]

                ra, rq, rz, rt, rsk, rku = compute_roughness_params(profile["Roughness"])
                r_l, rdq = compute_profile_roughness_ratio(profile["Length_mm"], profile["Roughness"])
                summary.update({"Ra_calc":ra,"Rq_calc":rq,"Rz_calc":rz,"Rt_calc":rt,"Rsk":rsk,"Rku":rku,"r_L":r_l,"Rdq":rdq})
                profile["Sample"] = sample_id
                profiles[uploaded.name] = profile
                summaries.append(summary)
            except Exception as error:
                st.error(f"Error processing {uploaded.name}: {error}")
        return standardize_roughness_columns(pd.DataFrame(summaries)), profiles


for key, default in {"master_df":pd.DataFrame(), "profile_dict":{}}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.markdown("### 1 · Data Input")
    with st.form("input_form", clear_on_submit=True):
        sample_name = st.text_input("Sample ID", "Sample A")
        files = st.file_uploader("Upload replicate files", type=["xlsx"], accept_multiple_files=True)
        filter_name = st.selectbox("Detrending filter", ["ISO Gaussian (λc)", "Savitzky-Golay", "None"])
        lambda_c, sg_window = 0.8, 51
        if filter_name == "ISO Gaussian (λc)":
            lambda_c = st.number_input("Cutoff wavelength λc (mm)", min_value=0.1, value=0.8, step=0.1)
        elif filter_name == "Savitzky-Golay":
            sg_window = st.slider("S-G window", 5, 151, 51, step=2)
        submit = st.form_submit_button("Add Sample Batch", use_container_width=True)

    if submit:
        if not files:
            st.warning("Select at least one Excel file.")
        else:
            new_summary, new_profiles = RoughnessLoader().process_files(files, sample_name, filter_name, sg_window, lambda_c)
            st.session_state.master_df = standardize_roughness_columns(pd.concat([st.session_state.master_df, new_summary], ignore_index=True))
            st.session_state.profile_dict.update(new_profiles)
            st.success(f"Added {len(new_summary)} valid file(s).")

    if not st.session_state.master_df.empty:
        st.markdown("### 2 · Manage Data")
        samples = sorted(st.session_state.master_df["Sample"].dropna().unique(), key=natural_sort_key)
        selected_batch = st.selectbox("Delete batch", ["Select"] + samples)
        if st.button("Delete Batch", use_container_width=True) and selected_batch != "Select":
            removed = st.session_state.master_df.loc[st.session_state.master_df["Sample"] == selected_batch, "File"].tolist()
            st.session_state.master_df = st.session_state.master_df.loc[st.session_state.master_df["Sample"] != selected_batch].reset_index(drop=True)
            for filename in removed:
                st.session_state.profile_dict.pop(filename, None)
            st.rerun()

    if st.button("Reset Entire Study", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()

render_header()
df = standardize_roughness_columns(st.session_state.master_df)
st.session_state.master_df = df
profiles = st.session_state.profile_dict

if df.empty:
    st.info("Upload one or more profilometer Excel files from the sidebar to begin.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.markdown(metric_card("Batches", df["Sample"].nunique()), unsafe_allow_html=True)
k2.markdown(metric_card("Replicates", len(df), "files"), unsafe_allow_html=True)
k3.markdown(metric_card("Mean Ra", f"{df['Ra'].mean():.3f}", "µm"), unsafe_allow_html=True)
k4.markdown(metric_card("Mean rL", f"{df['r_L'].mean():.6f}"), unsafe_allow_html=True)

tabs = st.tabs(["📋 Dataset", "📉 Trends", "🎨 Profiles", "📈 PSD", "💧 Wenzel", "💾 Export", "📖 Methods"])

with tabs[0]:
    section_title("Dataset", "📋")
    st.dataframe(df, use_container_width=True, height=430)

with tabs[1]:
    section_title("Inter-Sample Trends", "📉")
    parameters = [c for c in ["Ra","Rq","Rz","Rt","Ra_calc","Rq_calc","Rz_calc","Rt_calc","r_L","Rdq"] if c in df.columns and df[c].notna().any()]
    if not parameters:
        st.warning("No numeric roughness parameters are available.")
    else:
        parameter = st.selectbox("Parameter", parameters)
        summary = df.groupby("Sample")[parameter].agg(mean="mean", std="std", count="count").reset_index()
        fig = go.Figure(go.Scatter(x=summary["Sample"], y=summary["mean"], mode="lines+markers", error_y=dict(type="data", array=summary["std"].fillna(0), visible=True), line=dict(color="#0b1120", width=2.5), marker=dict(color="#c9a84c", size=10)))
        fig.update_layout(template="plotly_white", width=850, height=520, xaxis_title="Sample ID", yaxis_title=parameter)
        st.plotly_chart(fig, use_container_width=False)

with tabs[2]:
    section_title("Filtered Roughness Profiles", "🎨")
    available = sorted(profiles.keys(), key=natural_sort_key)
    if not available:
        st.warning("No profile data are available.")
    else:
        filename = st.selectbox("Profile file", available, key="profile_file")
        profile = profiles[filename]
        fig = go.Figure(go.Scatter(x=profile["Length_mm"], y=profile["Roughness"], mode="lines", line=dict(color="#3a7bd5")))
        fig.update_layout(template="plotly_white", width=900, height=520, xaxis_title="Profile length (mm)", yaxis_title="Roughness amplitude (µm)")
        st.plotly_chart(fig, use_container_width=False)

with tabs[3]:
    section_title("Power Spectral Density", "📈")
    available = sorted(profiles.keys(), key=natural_sort_key)
    if not available:
        st.warning("No profile data are available.")
    else:
        filename = st.selectbox("Profile file", available, key="psd_file")
        profile = profiles[filename]
        signal, x = profile["Roughness"].to_numpy(), profile["Length_mm"].to_numpy()
        dx = float(np.median(np.diff(x)))
        freq = np.fft.rfftfreq(len(signal), d=dx)
        power = np.abs(np.fft.rfft(signal - np.mean(signal))) ** 2
        mask = (freq > 0) & (power > 0)
        fig = go.Figure(go.Scatter(x=freq[mask], y=power[mask], mode="lines", line=dict(color="#9b59b6")))
        fig.update_layout(template="plotly_white", width=900, height=520, xaxis_type="log", yaxis_type="log", xaxis_title="Spatial frequency (cycles/mm)", yaxis_title="Power density")
        st.plotly_chart(fig, use_container_width=False)

with tabs[4]:
    section_title("Profile-Based Wenzel Correction", "💧")
    st.latex(r"\theta_Y=\cos^{-1}\left(\frac{\cos\theta_W}{r_L}\right)")
    valid = df.loc[df["r_L"].notna() & (df["r_L"] >= 1)].copy()
    if valid.empty:
        st.warning("No valid profile roughness ratios are available.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            filename = st.selectbox("Surface profile file", sorted(valid["File"].tolist(), key=natural_sort_key), key="wenzel_file")
        with c2:
            theta_w = st.number_input("Measured apparent contact angle θW (°)", min_value=0.0, max_value=180.0, value=90.0, step=0.1, format="%.2f")
        selected = valid.loc[valid["File"] == filename].iloc[0]
        r_l, rdq = float(selected["r_L"]), float(selected["Rdq"])
        theta_y = wenzel_corrected_angle(theta_w, r_l)
        a, b, c, d = st.columns(4)
        a.markdown(metric_card("Sample", selected["Sample"]), unsafe_allow_html=True)
        b.markdown(metric_card("Profile Ratio rL", f"{r_l:.6f}"), unsafe_allow_html=True)
        c.markdown(metric_card("RMS Slope Rdq", f"{rdq:.6f}"), unsafe_allow_html=True)
        d.markdown(metric_card("Corrected θY", f"{theta_y:.2f}", "°"), unsafe_allow_html=True)
        result = pd.DataFrame({"Sample":[selected["Sample"]],"Profile_File":[filename],"Apparent_CA_deg":[theta_w],"Profile_Roughness_Ratio_rL":[r_l],"RMS_Profile_Slope_Rdq":[rdq],"Corrected_Young_CA_deg":[theta_y],"Correction_deg":[theta_y-theta_w]})
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button("Download Wenzel Result", export_excel(result, "Wenzel_Correction"), "Wenzel_Contact_Angle_Correction.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.warning("rL is a two-dimensional profile-length approximation. Apply each rL only to the surface represented by the same profile file.")

with tabs[5]:
    section_title("Export", "💾")
    st.download_button("Download Summary", export_excel(df, "Summary"), "SRoughnessLab_Summary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if profiles:
        blocks = []
        for filename in sorted(profiles, key=natural_sort_key):
            block = profiles[filename][["Length_mm","Amplitude_um","Roughness"]].reset_index(drop=True).copy()
            stem = os.path.splitext(filename)[0]
            block.columns = [f"{stem}_X_mm", f"{stem}_Amplitude_um", f"{stem}_Roughness_um"]
            blocks.append(block)
        st.download_button("Download All Profiles", export_excel(pd.concat(blocks, axis=1), "Profiles"), "SRoughnessLab_All_Profiles.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tabs[6]:
    section_title("Methods", "📖")
    st.markdown(r"""
The profile-length roughness ratio is calculated as

\[r_L=\frac{\sum\sqrt{(\Delta x)^2+(\Delta z)^2}}{\sum\Delta x}\]

where horizontal coordinates are converted from millimetres to micrometres before segment lengths are computed. The RMS profile slope is

\[R_{dq}=\sqrt{\mathrm{mean}\left[\left(\frac{\Delta z}{\Delta x}\right)^2\right]}\]

The Wenzel-adjusted angle is calculated as

\[\theta_Y=\cos^{-1}\left(\frac{\cos\theta_W}{r_L}\right)\]

The reported rL is a two-dimensional profile-based approximation, not a direct three-dimensional area ratio.
""")
