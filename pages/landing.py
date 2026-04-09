import streamlit as st
import tempfile
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from process.data_extraction import process_eeg_for_inference
from process.deniz import fix_eeg_channels_load_edf, load_edf_file
from process.EEGPreprocessor import EEGPreprocessor
from process.visualization import plot_interactive_eeg, plot_window_predictions
from process.ensemble_inference import (
    get_model_paths_grouped,
    run_decision_fusion_inference,
    window_level_align,
    calculate_metrics,
    soft_ensemble,
    hard_vote_ensemble
)

# ─── Constants ────────────────────────────────────────────────────────────────
FINAL_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ',  'CZ-PZ',
]
THRESHOLD = 0.5
REFERENCE_WS = 0.5  # We align all arrays to the 0.5s window timeline
LARGEST_WINDOW_REF = 10.0

def generate_notepad_content(file_name, ens_classes, window_size=0.5):
    """Converts prediction classes back to continuous Seizure Events."""
    in_seizure = False
    start_sec = 0
    seizures = []
    
    for i, c in enumerate(ens_classes):
        if c == 1 and not in_seizure:
            in_seizure = True
            start_sec = int(i * window_size)
        elif c == 0 and in_seizure:
            in_seizure = False
            end_sec = int(i * window_size)
            seizures.append((start_sec, end_sec))
    if in_seizure:
        seizures.append((start_sec, int(len(ens_classes) * window_size)))

    lines = []
    lines.append(f"File Name: {file_name}")
    lines.append(f"Number of Seizures in File: {len(seizures)}")
    for (start, end) in seizures:
        lines.append(f"Seizure Start Time: {start} seconds")
        lines.append(f"Seizure End Time: {end} seconds")
    return "\n".join(lines)


# ─── Styling ──────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="EEG Analyzer", page_icon="🧠")
st.markdown("""
<style>
/* Global */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0a0a1a 0%, #12122a 50%, #0d1b2a 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }
.main-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}
.subtitle { color: rgba(255,255,255,0.45); font-size: 0.95rem; letter-spacing: 0.08em; margin-bottom: 1.5rem; }
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 12px 16px;
    backdrop-filter: blur(8px);
}
.verdict-seizure { background: linear-gradient(90deg, rgba(239,68,68,0.25), rgba(239,68,68,0.05)); border: 1px solid rgba(239,68,68,0.6); border-radius: 12px; padding: 14px 20px; font-size: 1.1rem; font-weight: 700; color: #FCA5A5; margin-bottom: 1rem; }
.verdict-safe { background: linear-gradient(90deg, rgba(52,211,153,0.2), rgba(52,211,153,0.04)); border: 1px solid rgba(52,211,153,0.5); border-radius: 12px; padding: 14px 20px; font-size: 1.1rem; font-weight: 700; color: #6EE7B7; margin-bottom: 1rem; }
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🧠 Advanced EEG Seizure Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Multi-Resolution Ensemble · Dynamic Evaluation against Ground Truth<br>'
    '6 Window Sizes (0.5s | 1.0s | 2.0s | 4.0s | 5.0.s | 10.0s)</p>',
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1.15, 1], gap="large")

# ==============================================================================
# LEFT COLUMN — Data Processing
# ==============================================================================
with col_left:
    st.markdown("## 📂 Data Input & Visualization")
    
    st.markdown("**(Optional) Ground Truth Summary File**")
    summary_file_buffer = st.file_uploader(
        "Upload summary file (e.g., sample_seizured.txt) to automatically enable metrics Evaluation.", 
        type=['txt']
    )
    
    summary_path = None
    if summary_file_buffer:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as ts:
            ts.write(summary_file_buffer.getvalue())
            summary_path = ts.name
    elif os.path.exists("sample_seizured.txt"):
        summary_path = os.path.abspath("sample_seizured.txt")
        st.caption("ℹ️ Local `sample_seizured.txt` successfully found and loaded as default ground truth.")

    uploaded_edf = st.file_uploader(
        "Drop an EEG .edf file here",
        type=['edf'],
        help="Supports any CHB-MIT format .edf file with the standard 18-channel bipolar montage.",
    )

    if uploaded_edf is not None:
        temp_dir_edf = tempfile.mkdtemp()
        temp_edf_path = os.path.join(temp_dir_edf, uploaded_edf.name)
        with open(temp_edf_path, "wb") as f:
            f.write(uploaded_edf.getvalue())

        # Reset session state on new file
        if st.session_state.get('last_file') != uploaded_edf.name:
            st.session_state.pop('results', None)
            st.session_state.pop('y_true_ref', None)
        
        st.session_state['temp_edf_path'] = temp_edf_path
        st.session_state['summary_path'] = summary_path
        st.session_state['last_file'] = uploaded_edf.name

        st.success(f"✅ **{uploaded_edf.name}** ready.")

        with st.spinner("🔄 Rendering raw signals..."):
            raw_signals, ch_names, fs = fix_eeg_channels_load_edf(
                temp_edf_path, FINAL_CHANNELS, load_edf_file
            )
            
            if raw_signals is not None:
                duration_s = raw_signals.shape[1] / fs
                ci1, ci2, ci3 = st.columns(3)
                ci1.metric("Duration", f"{duration_s:.0f} s")
                ci2.metric("Channels", raw_signals.shape[0])
                ci3.metric("Sampling Rate", f"{int(fs)} Hz")

                st.markdown("### 📊 Raw EEG")
                st.plotly_chart(plot_interactive_eeg(raw_signals, ch_names, fs, title="Raw EEG", duration_to_plot=20), use_container_width=True)

                st.markdown("### 🧹 Preprocessed EEG")
                preprocessor = EEGPreprocessor(sampling_rate=fs, target_sfreq=128)
                proc_signals = preprocessor.preprocess(raw_signals)
                st.plotly_chart(plot_interactive_eeg(proc_signals, ch_names, 128, title="Preprocessed (Bandpass, Notch, Z-Score)", duration_to_plot=20), use_container_width=True)
            else:
                st.error("Failed to read EDF File.")

# ==============================================================================
# RIGHT COLUMN — Artificial Intelligence Pipeline
# ==============================================================================
with col_right:
    st.markdown("## 🤖 Inference Engine & Report")
    
    if 'temp_edf_path' not in st.session_state:
        st.info("👈 Please initialize the pipeline by uploading an `.edf` file.")
    else:
        models_dict = get_model_paths_grouped()
        if not models_dict:
            st.error("⚠️ No models detected inside `model_paths/`.")
        else:
            cm1, cm2 = st.columns(2)
            cm1.metric("Unique Window Sizes", len(models_dict))
            cm2.metric("Total Model Folds", sum(len(v) for v in models_dict.values()))
            
            if st.button("▶ Run Full Stack Ensemble Inference", type="primary", use_container_width=True):
                group_results = {}
                progress = st.progress(0, text="Initializing Ensemble...")
                out_dir = os.path.join(tempfile.gettempdir(), "eeg_out")
                
                # Sort robustly by identifying digits in folder name (e.g. 0.5s, 1.0s)
                sorted_groups = sorted(
                    models_dict.items(),
                    key=lambda x: float(re.search(r'_([\d.]+)s', x[0]).group(1)) if re.search(r'_([\d.]+)s', x[0]) else 99.0
                )

                # Identify the reference run (0.5s) to establish target y_length
                ref_group = sorted_groups[0]
                ws_m = re.search(r'_([\d.]+)s', ref_group[0])
                ref_ws = float(ws_m.group(1)) if ws_m else 0.5
                
                y_true_reference = None
                target_length = None

                for step, (group_name, paths) in enumerate(sorted_groups):
                    ws_m = re.search(r'_([\d.]+)s', group_name)
                    ws = float(ws_m.group(1)) if ws_m else 10.0
                    label = f"{ws}s"
                    
                    progress.progress((step+1) / (len(sorted_groups)+1), text=f"Processing Network: {label} ({step+1}/{len(sorted_groups)})")
                    
                    try:
                        res = process_eeg_for_inference(
                            file_path=st.session_state['temp_edf_path'],
                            summary_file=st.session_state['summary_path'],
                            output_dir=out_dir,
                            window_size=ws,
                            largest_window_ref=LARGEST_WINDOW_REF,
                            overlap=0.0
                        )
                        X_ws = res['X']
                        y_ws = res['y']
                        
                        # Store reference Y if this is the reference (usually 0.5s)
                        if getattr(st.session_state, 'target_len_captured', False) == False and ws == REFERENCE_WS:
                            y_true_reference = y_ws
                            target_length = len(y_ws)
                            st.session_state['y_true_ref'] = y_true_reference
                            st.session_state.target_len_captured = True
                        
                        probs, classes = run_decision_fusion_inference(X_ws, {group_name: paths})
                        
                        group_results[group_name] = {
                            'ws': ws,
                            'label': label,
                            'probs_raw': probs,
                            'classes_raw': classes,
                        }
                    except Exception as e:
                        st.warning(f"Error processing {label}: {e}")
                
                # --- Time Alignment & Ensemble Processing ---
                progress.progress(0.95, text="Aligning dimensions and executing Voting Algorithms...")
                
                # Fallback target_length just in case 0.5s didn't run properly
                if target_length is None and 'y_true_ref' in st.session_state:
                    y_true_reference = st.session_state['y_true_ref']
                    target_length = len(y_true_reference)
                if target_length is None:
                    y_true_reference = np.zeros(0) # Failsafe
                    target_length = 1 
                
                aligned_probs_list = []
                metrics_list = []
                has_labels = len(np.unique(y_true_reference)) > 1

                for gn, gr in group_results.items():
                    aligned = window_level_align(gr['probs_raw'], target_length)
                    gr['probs'] = aligned
                    gr['classes'] = (aligned >= THRESHOLD).astype(int)
                    aligned_probs_list.append(aligned)
                    
                    if has_labels:
                        m = calculate_metrics(y_true_reference, aligned, THRESHOLD)
                        m['Model'] = f"{gr['ws']}s Model"
                        metrics_list.append(m)
                
                if aligned_probs_list:
                    soft_vote_probs = soft_ensemble(aligned_probs_list)
                    soft_vote_classes = (soft_vote_probs >= THRESHOLD).astype(int)
                    
                    group_results['__ensemble__'] = {
                        'ws': REFERENCE_WS,
                        'label': '🏆 Soft Vote Ensemble',
                        'probs': soft_vote_probs,
                        'classes': soft_vote_classes
                    }
                    
                    if has_labels:
                        ms = calculate_metrics(y_true_reference, soft_vote_probs, THRESHOLD)
                        ms['Model'] = '🏆 Soft Vote Ensemble'
                        metrics_list.append(ms)

                progress.progress(1.0, text="✅ Diagnostics Complete.")
                st.session_state['results'] = group_results
                st.session_state['metrics_list'] = metrics_list
                st.session_state['has_labels'] = has_labels

            # --- RENDER RESULTS ---
            if 'results' in st.session_state:
                res = st.session_state['results']
                ens = res.get('__ensemble__')
                if not ens:
                    st.error("Ensemble resolution failed.")
                else:
                    seiz_count = sum(ens['classes'])
                    tot_count = len(ens['classes'])
                    sz_pct = (seiz_count / tot_count * 100) if tot_count > 0 else 0
                    
                    if seiz_count > 0:
                        st.markdown(f'<div class="verdict-seizure">⚠️ SEIZURE DETECTED &nbsp;·&nbsp; {sz_pct:.1f}% positive severity</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="verdict-safe">✅ NO SEIZURES DETECTED &nbsp;·&nbsp; Normal</div>', unsafe_allow_html=True)
                    
                    # TABS
                    t_timeline, t_heatmap, t_bar, t_report = st.tabs([
                        "📈 Timeline", "📊 Diagnostics Eval", "📊 Metrics Eval", "📋 Extracted Report"
                    ])

                    with t_timeline:
                        st.caption("Visualizing specific window prediction curves against final Fusion Ensemble.")
                        # plot_window_predictions in visualization.py expects 'mean_prob' and 'total'. Quick shim:
                        for gn in res.keys():
                            res[gn]['total'] = len(res[gn]['classes'])
                            res[gn]['seizure'] = sum(res[gn]['classes'])
                            res[gn]['mean_prob'] = float(np.mean(res[gn]['probs']))
                            
                        fig_tl = plot_window_predictions(res, threshold=THRESHOLD)
                        st.plotly_chart(fig_tl, use_container_width=True)

                    with t_heatmap:
                        if st.session_state.get('has_labels'):
                            df_m = pd.DataFrame(st.session_state['metrics_list'])
                            df_m = df_m.drop(columns=['Decision Threshold'], errors='ignore')
                            df_m.set_index('Model', inplace=True)
                            
                            fig, ax = plt.subplots(figsize=(10, 4))
                            sns.heatmap(df_m, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
                            ax.set_title("Ground Truth Evaluation Heatmap", fontsize=12)
                            sns.set_theme(style="whitegrid", context="paper")
                            fig.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.info("No ground truth labels could be matched from the summary text. Inference is running blindly.")
                            
                    with t_bar:
                        if st.session_state.get('has_labels'):
                            df_m = pd.DataFrame(st.session_state['metrics_list'])
                            df_m = df_m.drop(columns=['Decision Threshold'], errors='ignore')
                            df_m_melt = df_m.melt(id_vars='Model', var_name='Metric', value_name='Score')
                            fig_b, ax_b = plt.subplots(figsize=(10, 4))
                            sns.barplot(data=df_m_melt, x='Metric', y='Score', hue='Model', palette='viridis', ax=ax_b)
                            ax_b.set_ylim(0, 1.05)
                            ax_b.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
                            plt.xticks(rotation=15)
                            fig_b.tight_layout()
                            st.pyplot(fig_b)
                        else:
                            st.info("Unavailable under blind inference mode.")
                            
                    with t_report:
                        txt_report = generate_notepad_content(st.session_state['last_file'], ens['classes'], REFERENCE_WS)
                        st.text_area("Live Extracted Notation Data", txt_report, height=250)
                        
                        st.download_button(
                            label="⬇️ Download Output File",
                            data=txt_report,
                            file_name=f"{st.session_state['last_file']}_seizured.txt",
                            mime="text/plain",
                            type="secondary"
                        )