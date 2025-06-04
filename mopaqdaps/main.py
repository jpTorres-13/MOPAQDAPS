import os
import json
import argparse
import shutil
import math
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import librosa.display

import scipy.signal
from sklearn.preprocessing import StandardScaler

import torch
import torchaudio
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def spectrogram(y, sr, n_fft=1024, hop_length=1024):
    """
    Compute a dB‐scaled spectrogram from audio samples.
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    return librosa.amplitude_to_db(S, ref=np.max)


def analytical_metrics(audio, sr):
    """
    Compute analytical features on a single waveform:
      - Zero Crossing Rate
      - Audio Energy (sum of squares)
      - Spectrum Flatness Measure
      - Tonality (via HPSS)
      - Harmonic Ratio (mean of YIN f0 estimate)
      - Spectral Centroid
      - Mean Absolute Frequency Deviation (via centroid diffs)
      - Number of Sinusoidal Peaks (via spectrogram peaks)
    """
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    energy = np.sum(audio**2)
    sfm = np.mean(librosa.feature.spectral_flatness(y=audio))

    harmonic, _ = librosa.effects.hpss(audio)
    tonality = np.mean(np.abs(harmonic))

    harmonic_ratio = np.mean(librosa.yin(audio, fmin=librosa.note_to_hz('C0'), fmax=librosa.note_to_hz('C5')))

    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    frequencies = np.abs(np.diff(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    mafd = np.mean(frequencies)

    frequencies, times, spectrogram = scipy.signal.spectrogram(audio, sr)
    peaks = scipy.signal.find_peaks(np.mean(spectrogram,axis=1))[0]

    return {
        'Zero Crossing Rate': zcr,
        'Audio Energy': energy,
        'Spectrum Flatness Measure': sfm,
        'Tonality': tonality,
        'Harmonic Ratio': harmonic_ratio,
        'Spectral Centroid': centroid,
        'Mean Absolute Frequency Deviation': mafd,
        'Number of Sinusoidal Peaks': len(peaks)
    }


def comparative_metrics(a1, a2, sr, device=None):
    """
    Compute comparative features between two waveforms:
      - MFCC L1 Distance
      - Loudness Distance (RMS difference)
      - Detuning (mean |f1−f2| via PYIN)
      - Cross‐Correlation Max
      - Drift Compensation (mean phase‐difference scaled)
      - Phase Reset Distance (counts of phase resets)
    """
    # initialize and cache on first call (or if sr/device changes)
    if not hasattr(comparative_metrics, 'initialized') \
       or comparative_metrics.sr != sr \
       or (device and comparative_metrics.device != device):
        # set device
        dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        comparative_metrics.device = dev
        comparative_metrics.sr = sr
        # MFCC transform
        comparative_metrics.mfcc_tf = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=20,
            melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128, "window_fn": torch.hann_window}
        ).to(dev)
        # STFT window
        comparative_metrics.window = torch.hann_window(2048, device=dev)
        comparative_metrics.initialized = True

    dev = comparative_metrics.device
    mfcc_tf = comparative_metrics.mfcc_tf
    window = comparative_metrics.window

    # to tensor
    t1 = torch.as_tensor(a1, dtype=torch.float32, device=dev)
    t2 = torch.as_tensor(a2, dtype=torch.float32, device=dev)

    # align length
    n = min(t1.numel(), t2.numel())
    t1, t2 = t1[:n], t2[:n]

    # batched STFT for RMS & phase
    batch = torch.stack([t1, t2], dim=0)
    st = torch.stft(
        batch,
        n_fft=2048,
        hop_length=512,
        window=window,
        return_complex=True,
        center=True,
        pad_mode='reflect'
    )  # shape (2, freq, frames)
    mag2 = st.abs() ** 2
    # frame‑wise RMS
    rms = torch.sqrt(mag2.mean(dim=1))  # (2, frames)
    # phase
    phase = torch.angle(st)

    # MFCC (batch)
    mf = mfcc_tf(batch.unsqueeze(1))  # (2, n_mfcc, frames)
    m = mf.shape[-1]
    mfcc_dist = float((mf[0,:,:m] - mf[1,:,:m]).abs().mean().item())

    # loudness distance
    loud_dist = float((rms[0] - rms[1]).abs().mean().item())

    # detune: CPU fallback
    x1 = t1.detach().cpu().numpy()
    x2 = t2.detach().cpu().numpy()
    p1, _, _ = librosa.pyin(x1, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    p2, _, _ = librosa.pyin(x2, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    m3 = min(len(p1), len(p2))
    detune = np.nanmean(np.abs(p1[:m3] - p2[:m3]))

    # cross‑correlation via FFT (full mode)
    L = x1.size + x2.size - 1
    fsize = 1 << math.ceil(math.log2(L))
    A = torch.fft.rfft(t1, n=fsize)
    B = torch.fft.rfft(t2, n=fsize)
    cc = torch.fft.irfft(A * torch.conj(B), n=fsize)
    cc_max = float(cc.abs().max().item())

    # phase drift
    ph1, ph2 = phase[0], phase[1]
    mf2 = min(ph1.shape[1], ph2.shape[1])
    drift = float((ph1[:,:mf2] - ph2[:,:mf2]).abs().mean().item())
    drift = drift * ph1.numel() / ph1.shape[1] #drift / (ph1.numel())

    # phase reset distance
    def resets(sig):
        ana = scipy.signal.hilbert(sig)
        ip = np.unwrap(np.angle(ana))
        return np.where(np.diff(ip) > 0.1)[0] #np.where(np.diff(ip) > np.pi)[0]
    r1 = resets(x1)
    r2 = resets(x2)
    mr = abs(len(r1) - len(r2))>1 #min(len(r1), len(r2))
    pr_dist = float(np.mean(np.abs(r1[:mr] - r2[:mr]))) if mr > 0 else 0.0

    return {
        'MFCC L1 Distance':      mfcc_dist,
        'Loudness Distance':     loud_dist,
        'Detuning':              detune,
        'Cross-Correlation Max': cc_max,
        'Drift Compensation':    drift,
        'Phase Reset Distance':  pr_dist
    }


def make_json_serializable(x):
    """
    Utility to convert NumPy types (and nested dicts/lists) to built‐in Python types.
    """
    if isinstance(x, dict):
        return {k: make_json_serializable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [make_json_serializable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def analytical_evaluation(metric, reference_value, shifted_value, shift_ratio):
    """
    Compute a relative error for analytical metrics:
      - Some metrics should scale proportionally with shift_ratio (frequency‐related).
      - Some should stay constant (energy, flatness, etc.).
    """
    freq_metrics = {
        'Zero Crossing Rate',
        'Spectral Centroid',
        'Harmonic Ratio',
        'Mean Absolute Frequency Deviation'
    }
    const_metrics = {
        'Audio Energy',
        'Spectrum Flatness Measure',
        'Tonality',
        'Number of Sinusoidal Peaks'
    }

    def rel_error(obs, exp):
        if exp == 0:
            return 0.0
        return abs(obs - exp) / abs(exp)

    if metric in freq_metrics:
        expected = reference_value * shift_ratio
        return rel_error(shifted_value, expected)
    elif metric in const_metrics:
        return rel_error(shifted_value, reference_value)
    else:
        return float('nan')


def comparison_evaluation(metric, metric_value, shift_ratio, reference_pitch=None, reference_energy=None):
    """
    Compute a relative error for comparative metrics:
      - MFCC, Loudness, Drift, Phase Reset → reference is 0
      - Detuning → expected = |shift_ratio − 1| * reference_pitch
      - Cross‐Correlation Max → expected = reference_energy (i.e. max‐possible)
    """
    def rel_error(obs, exp):
        if exp == 0:
            return abs(obs)
        return abs(obs - exp) / abs(exp)

    if metric == 'MFCC L1 Distance':
        return abs(metric_value)
    elif metric == 'Loudness Distance':
        return abs(metric_value)
    elif metric == 'Detuning':
        if reference_pitch is None:
            raise ValueError("Detuning needs reference_pitch.")
        expected = abs(shift_ratio - 1.0) * reference_pitch
        return rel_error(metric_value, expected)
    elif metric == 'Cross-Correlation Max':
        if reference_energy is None:
            raise ValueError("Cross-Correlation Max needs reference_energy.")
        expected = reference_energy
        return rel_error(metric_value, expected)
    elif metric == 'Drift Compensation':
        return abs(metric_value)
    elif metric == 'Phase Reset Distance':
        return abs(metric_value)
    else:
        return float('nan')


def plot_spectrograms(input_dir, output_dir, spectrograms_dir, sr=16000, max_duration=20):
    """
    For each input .wav, plot its spectrogram alongside every shifted output
    from each plugin/ratio. Save each figure as {input_basename}_{plugin}_all_ratios.png
    into spectrograms_dir.
    """
    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.wav')])
    plugins = sorted([p for p in os.listdir(output_dir)
                      if os.path.isdir(os.path.join(output_dir, p))])
    all_ratios = sorted({
        r for p in plugins
        for r in os.listdir(os.path.join(output_dir, p))
        if os.path.isdir(os.path.join(output_dir, p, r))
    })

    os.makedirs(spectrograms_dir, exist_ok=True)

    HOP = 1024
    N_FFT = 1024

    for inp in input_files:
        inp_path = os.path.join(input_dir, inp)
        if not os.path.exists(inp_path):
            continue

        y_in, _ = librosa.load(inp_path, sr=sr, duration=max_duration)
        spec_in = spectrogram(y_in, sr, n_fft=N_FFT, hop_length=HOP)

        for plugin in plugins:
            ratios = all_ratios
            total = 1 + len(ratios)
            fig, axes = plt.subplots(1, total, figsize=(5 * total, 4), constrained_layout=True)

            # Plot input
            img_in = librosa.display.specshow(
                spec_in, sr=sr, hop_length=HOP,
                x_axis='time', y_axis='log', ax=axes[0]
            )
            axes[0].set_title(f'Input\n{inp}')
            axes[0].set_yticks([])

            color_ref = None
            for i, ratio in enumerate(ratios):
                ax = axes[i + 1]
                out_path = os.path.join(output_dir, plugin, ratio, inp)
                if os.path.exists(out_path):
                    y_out, _ = librosa.load(out_path, sr=sr, duration=max_duration)
                    spec_out = spectrogram(y_out, sr, n_fft=N_FFT, hop_length=HOP)
                    img = librosa.display.specshow(
                        spec_out, sr=sr, hop_length=HOP,
                        x_axis='time', y_axis='log', ax=ax
                    )
                    ax.set_title(f'{plugin} | {ratio}')
                    color_ref = img
                else:
                    ax.text(0.5, 0.5, f'Missing:\n{plugin}\n{ratio}',
                            ha='center', va='center', fontsize=12)
                    ax.axis('off')
                ax.set_yticks([])

            if color_ref is not None:
                fig.colorbar(
                    color_ref,
                    ax=axes,
                    location='right',
                    format='%+2.0f dB',
                    pad=0.02,
                    shrink=0.9
                )

            out_fname = f"{os.path.splitext(inp)[0]}_{plugin}_all_ratios.png"
            plt.savefig(os.path.join(spectrograms_dir, out_fname), dpi=120)
            plt.close(fig)
            del fig, axes
            gc.collect()

        del y_in, spec_in
        gc.collect()


def calculate_metrics(input_dir, output_dir, analytical_dir, comparison_dir, results_dir):
    """
    1. Iterate over every input file and every plugin/ratio to compute:
       - Analytical metrics (per‐audio feature vector)
       - Comparative metrics (original vs. shifted)
    2. Save per‐input JSONs under analytical_dir and comparison_dir.
    3. Save per‐plugin JSONs and summary CSVs under results_dir.
    """
    # --------------- Prepare directories ---------------
    os.makedirs(analytical_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.wav')])
    plugins = sorted([p for p in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, p))])
    all_ratios = sorted({
        r for p in plugins
        for r in os.listdir(os.path.join(output_dir, p))
        if os.path.isdir(os.path.join(output_dir, p, r))
    })

    # --------------- Original metrics ---------------
    originals_metrics = {}
    for inp in input_files:
        audio, sr = librosa.load(os.path.join(input_dir, inp), sr=None)
        orig_anal = analytical_metrics(audio, sr)
        orig_comp = comparative_metrics(audio, audio, sr)
        originals_metrics[inp] = {'analytical': orig_anal, 'comparison': orig_comp}

    # Ensure per-input subdirectories
    for inp in input_files:
        base = os.path.splitext(inp)[0]
        os.makedirs(os.path.join(analytical_dir, base), exist_ok=True)
        os.makedirs(os.path.join(comparison_dir, base), exist_ok=True)

    # --------------- Per‐plugin results data structure ---------------
    plugins_results = {pl: [] for pl in plugins}

    # --------------- Main loop: input × ratio × plugin ---------------
    for inp in input_files:
        base = os.path.splitext(inp)[0]
        audio_in, sr_in = librosa.load(os.path.join(input_dir, inp), sr=None)
        orig_anal = originals_metrics[inp]['analytical']

        for ratio in all_ratios:
            analytical_dict = {
                'input_file': inp,
                'pitch_ratio': ratio,
                'original': orig_anal,
                'shifted': {},
                'normalized': {}
            }
            comparison_dict = {
                'input_file': inp,
                'pitch_ratio': ratio,
                'original': originals_metrics[inp]['comparison'],
                'shifted': {}
            }

            for plugin in plugins:
                shifted_path = os.path.join(output_dir, plugin, ratio, inp)
                if not os.path.exists(shifted_path):
                    continue

                y_out, sr_out = librosa.load(shifted_path, sr=None)
                analy_res = analytical_metrics(y_out, sr_out)
                analytical_dict['shifted'][plugin] = analy_res
                normalized = {}
                for k, orig_val in orig_anal.items():
                    norm_val = analy_res[k] / orig_val if orig_val != 0 else 0
                    normalized[k] = norm_val
                analytical_dict['normalized'][plugin] = normalized

                comp_res = comparative_metrics(audio_in, y_out, sr_in)
                comparison_dict['shifted'][plugin] = comp_res

                plugins_results[plugin].append({
                    'input_file': inp,
                    'pitch_ratio': ratio,
                    'analytical': analy_res,
                    'comparison': comp_res
                })

            # Save per‐input JSONs
            with open(os.path.join(analytical_dir, base, f"{ratio}.json"), 'w') as f:
                json.dump(make_json_serializable(analytical_dict), f, indent=2)
            with open(os.path.join(comparison_dir, base, f"{ratio}.json"), 'w') as f:
                json.dump(make_json_serializable(comparison_dict), f, indent=2)

    # --------------- Save per‐plugin JSONs ---------------
    for plugin, recs in plugins_results.items():
        with open(os.path.join(results_dir, f"{plugin}.json"), 'w') as f:
            json.dump(make_json_serializable(recs), f, indent=2)

    # --------------- Save original metrics summary ---------------
    with open(os.path.join(results_dir, 'original.json'), 'w') as f:
        json.dump(make_json_serializable(originals_metrics), f, indent=2)

    # --------------- Generate DataFrame & CSV for original metrics ---------------
    records = []
    for inp, data in originals_metrics.items():
        for domain, metrics in data.items():
            for metric_name, value in metrics.items():
                records.append({'input': inp, 'metric': metric_name, 'value': value})
    df_orig = pd.DataFrame(records)
    df_orig.to_csv(os.path.join(results_dir, 'original_metrics.csv'), index=False)

    # --------------- Compute z‐score summary for original metrics ---------------
    agg = df_orig.groupby(['metric']).agg(
        avg=('value', 'mean'),
        median=('value', 'median'),
        std=('value', 'std'),
        metric_count=('value', 'count')
    ).reset_index()

    temp_z = agg[['avg', 'std', 'median', 'metric_count']].copy()
    temp_z['metric_count'] *= -1  # invert to make “lower is better”
    z_scores = temp_z.apply(lambda col: (col - col.mean()) / col.std(ddof=0))
    agg['zscore'] = z_scores.mean(axis=1)

    metrics_summary = {}
    for _, row in agg.iterrows():
        metrics_summary[row['metric']] = {
            'avg': row['avg'],
            'median': row['median'],
            'std': row['std'],
            'metric_count': int(row['metric_count']),
            'zscore': row['zscore']
        }

    with open(os.path.join(results_dir, 'original_metrics_zscore.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    # --------------- Build plugin‐wise mean summaries ---------------
    plugin_means = {}
    for plugin in plugins:
        plugin_json = os.path.join(results_dir, f'{plugin}.json')
        with open(plugin_json, 'r') as f:
            data = json.load(f)

        records = []
        for rec in data:
            for domain in ['analytical', 'comparison']:
                for metric, val in rec[domain].items():
                    records.append({'metric': metric, 'value': val})
        dfp = pd.DataFrame(records)
        if not dfp.empty:
            agg_p = dfp.groupby('metric')['value'].mean().reset_index()
            plugin_means[plugin] = dict(zip(agg_p['metric'], agg_p['value']))
        else:
            plugin_means[plugin] = {}

    # --------------- Build comparison table CSV ---------------
    all_metrics = sorted(metrics_summary.keys())
    rows = []
    for m in all_metrics:
        row = {'metric': m, 'original_mean': metrics_summary[m]['avg']}
        for plugin in plugins:
            row[f'{plugin}_mean'] = plugin_means.get(plugin, {}).get(m, np.nan)
        rows.append(row)

    df_means = pd.DataFrame(rows)
    df_means.to_csv(os.path.join(results_dir, 'metrics_mean_comparison.csv'), index=False)

    return originals_metrics, plugins_results


def evaluate_errors(originals_metrics, plugins_results, analytical_dir, comparison_dir, results_dir):
    """
    1. For each input × ratio × plugin:
       - Compute analytical‐error = analytical_evaluation(...)
       - Compute comparative‐error = comparison_evaluation(...)
       - Store in a flat record list
    2. Output:
       - results_detailed.csv (all records)
       - JSON files (aggregated z‐score per plugin, per metric, per ratio, etc.)
       - Plotting of various summary graphs
       - Determine “best” plugin by composite zscore
    """
    # Collect records
    records = []
    input_files = sorted(originals_metrics.keys())
    plugins = sorted(plugins_results.keys())
    all_ratios = sorted({
        rec['pitch_ratio']
        for recs in plugins_results.values() for rec in recs
    })

    # We already have originals_metrics[input]['analytical'] and ['comparison']
    for inp in input_files:
        base = os.path.splitext(inp)[0]
        ref_anal = originals_metrics[inp]['analytical']
        ref_comp = originals_metrics[inp]['comparison']
        ref_energy = ref_anal['Audio Energy']
        ref_pitch = 0.0  # you might compute this from original audio if needed

        for ratio in all_ratios:
            anal_path = os.path.join(analytical_dir, base, f"{ratio}.json")
            comp_path = os.path.join(comparison_dir, base, f"{ratio}.json")
            if not os.path.exists(anal_path) or not os.path.exists(comp_path):
                continue

            with open(anal_path, 'r') as f:
                anal_json = json.load(f)
            with open(comp_path, 'r') as f:
                comp_json = json.load(f)

            for plugin in plugins:
                if plugin not in anal_json['shifted'] or plugin not in comp_json['shifted']:
                    continue

                # Analytical metrics
                for metric, shifted_val in anal_json['shifted'][plugin].items():
                    err = analytical_evaluation(
                        metric,
                        ref_anal[metric],
                        shifted_val,
                        float(ratio[5:])  # assuming ratio strings like "ratio-2.0"
                    )
                    records.append({
                        'plugin': plugin,
                        'input': base,
                        'ratio': ratio,
                        'metric': metric,
                        'error': err
                    })

                # Comparative metrics
                for metric, comp_val in comp_json['shifted'][plugin].items():
                    if metric == 'Detuning':
                        err = comparison_evaluation(
                            metric,
                            comp_val,
                            float(ratio[5:]),
                            reference_pitch=ref_pitch
                        )
                    elif metric == 'Cross-Correlation Max':
                        err = comparison_evaluation(
                            metric,
                            comp_val,
                            float(ratio[5:]),
                            reference_energy=ref_energy
                        )
                    else:
                        err = comparison_evaluation(metric, comp_val, float(ratio[5:]))
                    records.append({
                        'plugin': plugin,
                        'input': base,
                        'ratio': ratio,
                        'metric': metric,
                        'error': err
                    })

    # Create DataFrame & save detailed CSV
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(results_dir, 'results_detailed.csv'), index=False)

    # Define aggregation levels
    collections = {
        'metric_error_per_plugin': ['plugin', 'metric'],
        'metric_error_per_plugin_input': ['plugin', 'input', 'metric'],
        'metric_error_per_plugin_ratio': ['plugin', 'ratio', 'metric'],
        'metric_error_per_plugin_input_ratio': ['plugin', 'input', 'ratio', 'metric']
    }

    dataframes = {}
    for key, groupby_cols in collections.items():
        agg_errors = df.groupby(groupby_cols).agg(
            avg_error=('error', 'mean'),
            median_error=('error', 'median'),
            std_error=('error', 'std'),
            metric_count=('error', 'count')
        ).reset_index()

        # Compute z‐scores
        temp_z = agg_errors[['avg_error', 'std_error', 'median_error', 'metric_count']].copy()
        temp_z['metric_count'] *= -1
        z_scores = temp_z.apply(lambda col: (col - col.mean()) / col.std(ddof=0))
        agg_errors['zscore'] = z_scores.mean(axis=1)

        # Build nested JSON
        errors_dict = {}
        for _, row in agg_errors.iterrows():
            current = errors_dict
            for col in groupby_cols[:-1]:
                val = row[col]
                current = current.setdefault(val, {})
            leaf_key = row[groupby_cols[-1]]
            current[leaf_key] = {
                'avg_error': row['avg_error'],
                'std_error': row['std_error'],
                'median_error': row['median_error'],
                'metric_count': int(row['metric_count']),
                'zscore': row['zscore']
            }

        with open(os.path.join(results_dir, f'{key}.json'), 'w', encoding='utf-8') as f:
            json.dump(errors_dict, f, indent=2)

        # Also keep the DataFrame for plotting later
        dataframes[key] = agg_errors

    # --------------- Plotting graphs ---------------

    # 1) Z‐score by Input for each Metric
    df_ip = dataframes['metric_error_per_plugin_input']
    metrics_map = {
        'Zero Crossing Rate': 'ZCR',
        'Audio Energy': 'Energy',
        'Spectrum Flatness Measure': 'SFM',
        'Tonality': 'Tonality',
        'Harmonic Ratio': 'Harmonic Ratio',
        'Spectral Centroid': 'Centroid',
        'Mean Absolute Frequency Deviation': 'MAFD',
        'Number of Sinusoidal Peaks': 'Sinusoidal Peaks',
        'MFCC L1 Distance': 'MFCC Distance',
        'Loudness Distance': 'Loudness Dist',
        'Detuning': 'Detuning',
        'Cross-Correlation Max': 'Cross‐Corr',
        'Drift Compensation': 'Drift',
        'Phase Reset Distance': 'Phase Reset'
    }

    plot_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Z‐score by Input per Metric
    for metric in df_ip['metric'].unique():
        sub = df_ip[df_ip['metric'] == metric]
        pivot = sub.pivot(index='plugin', columns='input', values='zscore')
        plt.figure(figsize=(8, 5))
        for pl in pivot.index:
            plt.plot(
                pivot.columns,
                pivot.loc[pl],
                marker='o',
                label=pl
            )
        plt.xlabel('Input File')
        plt.ylabel('Z‐score')
        plt.title(f'Z‐score by Input for {metrics_map.get(metric, metric)}')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xticks(rotation=30, ha='right')
        plt.legend(fontsize=8, loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'zscore_by_input_{metric}.png'))
        plt.close()

    # Z‐score by Ratio for each Metric
    df_pr = dataframes['metric_error_per_plugin_ratio']
    for metric in df_pr['metric'].unique():
        sub = df_pr[df_pr['metric'] == metric]
        pivot = sub.pivot(index='plugin', columns='ratio', values='zscore')
        plt.figure(figsize=(8, 5))
        for pl in pivot.index:
            plt.plot(
                pivot.columns,
                pivot.loc[pl],
                marker='o',
                label=pl
            )
        plt.xlabel('Shift Ratio')
        plt.ylabel('Z‐score')
        plt.title(f'Z‐score by Ratio for {metrics_map.get(metric, metric)}')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xticks(rotation=30, ha='right')
        plt.legend(fontsize=8, loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'zscore_by_ratio_{metric}.png'))
        plt.close()

    # Boxplot of Z‐score per Plugin
    df_mp = dataframes['metric_error_per_plugin']
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_mp, x='plugin', y='zscore')
    plt.title("Z‐score Distribution by Plugin")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'zscore_distribution_per_plugin.png'))
    plt.close()

    # Best plugin per input (count of being “best”)
    best_plugins = df_ip.groupby(['input', 'plugin'])['zscore'].mean().reset_index()
    best_per_input = best_plugins.loc[best_plugins.groupby('input')['zscore'].idxmin()]
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=best_per_input,
        x='plugin',
        order=best_per_input['plugin'].value_counts().index
    )
    plt.title('Best Plugin by Input File (Count)')
    plt.ylabel('Count of Inputs')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'best_plugin_per_input_count.png'))
    plt.close()

    # Composite z‐score ranking per plugin
    plugin_summary = df_mp.groupby('plugin').agg(
        zscore=('zscore', 'mean'),
        avg_error=('avg_error', 'mean'),
        std_error=('std_error', 'mean'),
        metric_count=('metric_count', 'sum')
    ).sort_values('zscore')

    # Save ranking to a CSV
    plugin_summary.reset_index().to_csv(
        os.path.join(results_dir, 'plugin_composite_ranking.csv'), index=False
    )

    # Barplot of composite z‐score
    plt.figure(figsize=(10, 3))
    sns.barplot(
        data=plugin_summary.reset_index(),
        x='zscore', y='plugin',
        hue='plugin',
        palette='viridis',
        orient='h',
        legend=False
    )
    plt.title('Composite Z‐score Ranking by Plugin')
    plt.xlabel('Composite Z‐score')
    plt.ylabel('Plugin')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'composite_score_ranking.png'))
    plt.close()

    # ------------- Return plugin_summary so CLI can print it -------------
    return plugin_summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pitch‐shifting algorithms via analytical/comparative metrics."
    )
    parser.add_argument(
        "--dataset_root", "-d", required=True,
        help="Root folder containing subfolders 'input/' and 'output/'."
    )
    parser.add_argument(
        "--results_root", "-r", default="results",
        help="Where to write all outputs (JSON, CSV, PNG)."
    )
    parser.add_argument(
        "--skip_spectrograms", action="store_true",
        help="Skip plotting spectrograms if already generated."
    )
    args = parser.parse_args()

    # Define directories
    data_root = args.dataset_root
    input_dir = os.path.join(data_root, "input")
    output_dir = os.path.join(data_root, "output")
    spectrograms_dir = os.path.join(args.results_root, "spectrograms")
    analytical_dir = os.path.join(args.results_root, "metrics", "analytical")
    comparison_dir = os.path.join(args.results_root, "metrics", "comparison")
    results_dir = os.path.join(args.results_root, "results")

    # Create top‐level folders
    os.makedirs(args.results_root, exist_ok=True)
    os.makedirs(os.path.join(args.results_root, "metrics"), exist_ok=True)

    # 1) Plot spectrograms (unless skipped)
    if not args.skip_spectrograms:
        print(">>> Generating spectrogram plots...")
        plot_spectrograms(input_dir, output_dir, spectrograms_dir)

    # 2) Calculate metrics and save JSON/CSV
    print(">>> Calculating analytical & comparative metrics...")
    originals_metrics, plugins_results = calculate_metrics(
        input_dir, output_dir, analytical_dir, comparison_dir, results_dir
    )

    # 3) Evaluate errors and generate summary plots
    print(">>> Evaluating errors, computing z‐scores, plotting summaries...")
    plugin_summary = evaluate_errors(
        originals_metrics, plugins_results, analytical_dir, comparison_dir, results_dir
    )

    # 4) Print final ranking
    print("\n=== Composite Plugin Ranking (Lower z‐score = Better) ===")
    for i, (pl, row) in enumerate(plugin_summary.iterrows(), start=1):
        print(f"{i:2d}. {pl:15s} → score = {row['zscore']:.6f}")

    best = plugin_summary.index[0]
    best_score = plugin_summary.iloc[0]['zscore']
    print(f"\n🏆  Best plugin (composite score): {best}  (z‐score = {best_score:.6f})")


if __name__ == "__main__":
    main()
