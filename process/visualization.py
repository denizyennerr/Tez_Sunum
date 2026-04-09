import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def plot_interactive_eeg(signals, ch_names, fs, title="EEG Signal", duration_to_plot=10):
    """Plotly ile interaktif, sağa-sola kaydırılabilir EEG grafiği"""
    samples_to_plot = int(fs * duration_to_plot)
    if signals.shape[1] > samples_to_plot:
        signals_plot = signals[:, :samples_to_plot]
    else:
        signals_plot = signals

    time_axis = np.arange(signals_plot.shape[1]) / fs

    fig = go.Figure()
    offset_step = 3.0

    for i, ch_name in enumerate(ch_names):
        offset = i * offset_step
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=signals_plot[i] + offset,
            mode='lines',
            name=ch_name,
            line=dict(width=1)
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Time (Seconds)',
        yaxis=dict(showticklabels=False, title='Channels'),
        height=300,
        hovermode='x unified',
        xaxis=dict(rangeslider=dict(visible=True))
    )
    return fig


def get_ensemble_metrics_plots(csv_path="saved_outputs_play/ensemble_results_final/decision_fusion_macro_mean.csv"):
    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path)
    columns_to_drop = ['Decision Threshold', 'N_Subjects']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    df_metrics = df.set_index('Model')

    fig_heat, ax_heat = plt.subplots(figsize=(10, 2.5))
    sns.heatmap(df_metrics, annot=True, fmt=".4f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax_heat)
    ax_heat.set_title("Model Performance Heatmap")

    df_melted = df_metrics.reset_index().melt(id_vars='Model', var_name='Metric', value_name='Score')
    fig_bar, ax_bar = plt.subplots(figsize=(10, 2.5))
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', palette='viridis', ax=ax_bar)
    ax_bar.set_title("Metrics Comparison")
    ax_bar.set_ylim(0.0, 1.05)
    ax_bar.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    return fig_heat, fig_bar


# ─── NEW: Inference Visualizations ───────────────────────────────────────────

_MODEL_COLORS = [
    '#60A5FA',  # 0.5s  – sky blue
    '#34D399',  # 1.0s  – emerald
    '#FBBF24',  # 2.0s  – amber
    '#F87171',  # 4.0s  – red
    '#A78BFA',  # 5.0s  – violet
    '#FB923C',  # 10.0s – orange
]
_ENSEMBLE_COLOR = '#FFFFFF'


def plot_window_predictions(group_results: dict, threshold: float = 0.5):
    """
    window_size grubundaki her modelin tahmin olasılığını zaman ekseninde çizer.
    Ensemble ayrı, kalın beyaz çizgiyle gösterilir.
    group_results: { group_name: { 'ws', 'probs', 'classes', 'total', 'seizure', 'mean_prob', 'label' } }
    """
    fig = go.Figure()

    # Epoch-sıralı modeller (Ensemble sona)
    individual = [(gn, gr) for gn, gr in group_results.items() if gn != '__ensemble__']
    individual_sorted = sorted(individual, key=lambda x: x[1].get('ws') or 0)

    max_time = 0.0
    for color_idx, (gn, gr) in enumerate(individual_sorted):
        ws = gr.get('ws') or 10.0
        probs = np.array(gr['probs'])
        # Her pencerenin bitiş zamanını x ekseni olarak kullan
        time_axis = (np.arange(len(probs)) + 1) * ws
        max_time = max(max_time, float(time_axis[-1])) if len(time_axis) > 0 else max_time
        color = _MODEL_COLORS[color_idx % len(_MODEL_COLORS)]
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=probs,
            mode='lines',
            name=gr['label'],
            line=dict(width=1.5, color=color, dash='dot'),
            opacity=0.75,
        ))

    # Ensemble çizgisi
    if '__ensemble__' in group_results:
        ens = group_results['__ensemble__']
        ens_probs = np.array(ens['probs'])
        ens_ws = 10.0  # ensemble her zaman 10s referansla hizalanır
        ens_time = (np.arange(len(ens_probs)) + 1) * ens_ws
        max_time = max(max_time, float(ens_time[-1])) if len(ens_time) > 0 else max_time
        fig.add_trace(go.Scatter(
            x=ens_time,
            y=ens_probs,
            mode='lines',
            name='🏆 Ensemble (Soft Vote)',
            line=dict(width=3, color=_ENSEMBLE_COLOR),
        ))

    # Threshold çizgisi
    fig.add_shape(
        type='line', x0=0, x1=max_time, y0=threshold, y1=threshold,
        line=dict(color='rgba(239,68,68,0.8)', dash='dash', width=2)
    )
    fig.add_annotation(
        x=max_time, y=threshold, text=f" Threshold ({threshold})",
        showarrow=False, xanchor='right', font=dict(color='rgba(239,68,68,1)', size=11)
    )

    fig.update_layout(
        title=dict(text='Window-Level Seizure Probability — All Models vs Ensemble', font=dict(size=14)),
        xaxis_title='Time (seconds)',
        yaxis_title='Seizure Probability',
        yaxis=dict(range=[0, 1.05], gridcolor='rgba(255,255,255,0.1)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', rangeslider=dict(visible=True)),
        height=420,
        legend=dict(orientation='h', y=-0.35, font=dict(size=11)),
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,30,0.6)',
        margin=dict(t=50, b=100),
    )
    return fig


def plot_model_comparison_bar(group_results: dict):
    """
    Yatay bar chart: her modelin nöbet pencere yüzdesi ve ortalama olasılığı.
    """
    labels, seizure_pcts, mean_probs, bar_colors = [], [], [], []

    individual = [(gn, gr) for gn, gr in group_results.items() if gn != '__ensemble__']
    individual_sorted = sorted(individual, key=lambda x: x[1].get('ws') or 0)

    for color_idx, (gn, gr) in enumerate(individual_sorted):
        labels.append(f"{gr['ws']}s Model")
        pct = (gr['seizure'] / gr['total'] * 100) if gr['total'] > 0 else 0.0
        seizure_pcts.append(round(pct, 2))
        mean_probs.append(round(gr['mean_prob'], 4))
        bar_colors.append(_MODEL_COLORS[color_idx % len(_MODEL_COLORS)])

    if '__ensemble__' in group_results:
        ens = group_results['__ensemble__']
        labels.append('🏆 Ensemble (Soft Vote)')
        pct = (ens['seizure'] / ens['total'] * 100) if ens['total'] > 0 else 0.0
        seizure_pcts.append(round(pct, 2))
        mean_probs.append(round(ens['mean_prob'], 4))
        bar_colors.append('#FFD700')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=seizure_pcts,
        y=labels,
        orientation='h',
        marker=dict(color=bar_colors, line=dict(color='rgba(255,255,255,0.2)', width=1)),
        text=[f"{p:.1f}%  (avg prob: {m:.3f})" for p, m in zip(seizure_pcts, mean_probs)],
        textposition='outside',
        textfont=dict(size=12),
        name='Seizure Window %',
    ))

    x_max = max(seizure_pcts) * 1.3 + 5 if seizure_pcts else 10
    fig.update_layout(
        title=dict(text='Seizure Detection Rate — Model Comparison', font=dict(size=14)),
        xaxis=dict(title='Seizure Window %', range=[0, x_max], gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        height=360,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,15,30,0.6)',
        showlegend=False,
        margin=dict(l=160, r=20, t=50, b=40),
    )
    return fig