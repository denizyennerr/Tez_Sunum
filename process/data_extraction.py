import numpy as np
from process import deniz
from process.EEGPreprocessor import EEGPreprocessor
import os
import gc


def process_eeg_for_inference(
    file_path: str,
    summary_file: str,
    output_dir: str,
    window_size: float = 10.0,
    largest_window_ref: float = 10.0,
    overlap: float = 0.0,
    sampling_rate: int = 256,
    target_sfreq: int = 128,
) -> dict:
    """
    Yeni bir EEG kaydını inference için preprocess eder ve pencereler.

    Parameters
    ----------
    file_path        : İşlenecek .edf dosyasının tam yolu
    summary_file     : İlgili *-summary.txt dosyasının tam yolu
                       (seizure ground truth için; yoksa y tamamen 0 döner)
    output_dir       : Çıktı .npz dosyasının kaydedileceği dizin
    window_size      : Pencere boyutu (saniye)
    largest_window_ref: Zero-padding referansı (saniye) — eğitimde kullanılan
                        değerle aynı olmalı
    overlap          : Pencere örtüşme oranı (0.0 = sıfır örtüşme)
    sampling_rate    : Ham EEG örnekleme hızı
    target_sfreq     : Hedef örnekleme hızı (resample sonrası)

    Returns
    -------
    dict with keys:
        X         : np.ndarray, shape (n_windows, n_channels, n_samples)
        y         : np.ndarray, shape (n_windows,) — ground truth (varsa)
        file_ids  : list[str], her pencerenin kaynak dosyası
        output_path: kaydedilen .npz dosyasının yolu
    """

    FINAL_CHANNELS = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ'
    ]

    os.makedirs(output_dir, exist_ok=True)

    # --- Setup ---
    preprocessor = EEGPreprocessor(
        sampling_rate=sampling_rate,
        target_sfreq=target_sfreq
    )
    new_fs = preprocessor.target_sfreq
    window_samples = int(round(window_size * new_fs))
    step_samples = int(round(window_samples * (1 - overlap)))

    # --- Seizure annotation parse (opsiyonel ground truth) ---
    filename = os.path.basename(file_path)
    seizure_times = []

    if summary_file and os.path.exists(summary_file):
        seizure_info = deniz.parse_summary_file(summary_file)
        if seizure_info and filename in seizure_info:
            seizure_times = seizure_info[filename]
            print(f"📋 {len(seizure_times)} seizure annotation bulundu.")
        else:
            print("⚠️  Summary dosyasında bu dosyaya ait annotation bulunamadı. y=0 olarak işaretlenecek.")
    else:
        print("⚠️  Summary dosyası sağlanmadı. y=0 olarak işaretlenecek.")

    # --- Load & Preprocess ---
    print(f"📂 Yükleniyor: {file_path}")
    signals, ch_names, fs = deniz.fix_eeg_channels_load_edf(
        file_path=file_path,
        final_channels=FINAL_CHANNELS,
        load_func=deniz.load_edf_file
    )
    if signals is None:
        raise ValueError(f"EDF dosyası yüklenemedi: {file_path}")

    signals = preprocessor.preprocess(signals)
    n_samples = signals.shape[1]

    # --- Zero Padding ---
    duration_sec = n_samples / new_fs
    remainder = duration_sec % largest_window_ref
    if remainder > 0:
        pad_sec = largest_window_ref - remainder
        pad_samples = int(round(pad_sec * new_fs))
        padding = np.zeros((signals.shape[0], pad_samples), dtype=np.float32)
        signals = np.concatenate([signals, padding], axis=1)

    n_samples_padded = signals.shape[1]
    print(f"✅ Preprocess tamamlandı. Toplam süre: {n_samples_padded / new_fs:.1f}s")

    # --- Windowing ---
    X, y, file_ids = [], [], []
    seizure_count = 0

    for start in range(0, n_samples_padded - window_samples + 1, step_samples):
        end = start + window_samples
        window = signals[:, start:end]

        window_start_sec = start / new_fs
        window_end_sec = end / new_fs

        # -------------------------------------------------------------
        # ENSEMBLE ALIGNMENT RULE: NEVER DROP WINDOWS
        # -------------------------------------------------------------
        overlap_duration = 0.0
        for sz_start, sz_end in seizure_times:
            overlap_start = max(window_start_sec, sz_start)
            overlap_end = min(window_end_sec, sz_end)

            if overlap_end > overlap_start:
                overlap_duration += (overlap_end - overlap_start)

        overlap_ratio = overlap_duration / window_size

        # Kesin Karar: %50'dan büyük eşitse 1, değilse 0. Hiçbir pencere çöpe atılmaz.
        is_seizure = (overlap_ratio >= 0.50)

        X.append(window.astype(np.float32))
        y.append(1 if is_seizure else 0)
        file_ids.append(filename)

        if is_seizure:
            seizure_count += 1

    del signals
    gc.collect()

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    # Transpose X from (N, Channels, Window_Samples) to (N, Window_Samples, Channels)
    # This matches the model input shape expected (e.g. 64, 18 instead of 18, 64)
    if len(X) > 0:
        X = np.transpose(X, (0, 2, 1))

    print(f"📊 Toplam pencere: {len(X)} | Seizure: {seizure_count} | Normal: {len(X) - seizure_count}")

    # --- Save ---
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{base_name}_w{window_size}s.npz")
    np.savez_compressed(output_path, X=X, y=y, file_ids=file_ids)
    print(f"💾 Kaydedildi: {output_path}")

    return {
        "X": X,
        "y": y,
        "file_ids": file_ids,
        "output_path": output_path
    }