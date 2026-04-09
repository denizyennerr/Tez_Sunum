import glob
import re
from collections import defaultdict
import pyedflib
import numpy as np
import os


def costum():
    print('ww22ww2')

def load_edf_file(file_path):
    """
    Load an EDF file and return signals + metadata.
    """
    try:
        f = pyedflib.EdfReader(file_path)

        n_channels = f.signals_in_file
        channel_names = f.getSignalLabels()
        sampling_rate = int(f.getSampleFrequency(0))

        # Read all channels
        n_samples = f.getNSamples()[0]
        signals = np.zeros((n_channels, n_samples))

        for i in range(n_channels):
            signals[i, :] = f.readSignal(i)

        f.close()

        return signals, channel_names, sampling_rate

    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None, None


def parse_summary_file(summary_path):
    """
    Parse a patient's summary file to extract seizure times.

    Returns:
        dict: {filename: [(start, end), ...]}
    """
    seizure_info = {}

    with open(summary_path, 'r') as f:
        content = f.read()

    # Split by "File Name:" to get each file's info
    file_blocks = content.split('File Name:')[1:]  # Skip header

    for block in file_blocks:
        lines = block.strip().split('\n')

        # First line is the filename
        filename = lines[0].strip()

        # Find all seizure times in this block
        seizure_times = []
        start_time = None

        for line in lines:
            # Match any seizure start time pattern
            if 'Seizure' in line and 'Start Time' in line:
                try:
                    start_time = int(line.split(':')[-1].replace('seconds', '').strip())
                except:
                    pass

            # Match any seizure end time pattern
            if 'Seizure' in line and 'End Time' in line:
                try:
                    end_time = int(line.split(':')[-1].replace('seconds', '').strip())
                    if start_time is not None:
                        seizure_times.append((start_time, end_time))
                        start_time = None
                except:
                    pass

        if seizure_times:
            seizure_info[filename] = seizure_times

    return seizure_info


def fix_eeg_channels_load_edf(file_path, final_channels, load_func):
    """
    load_edf_file kullanarak:
    - Kanal isimlerini düzeltir
    - Eksik kanalları 0 ile doldurur
    - Kanal sırasını final_channels'a göre sabitler
    """

    try:
        signals, channel_names, fs = load_func(file_path)

        if signals is None:
            return None, None, None

        channel_names = list(channel_names)

        # -------------------------
        # 1️⃣ Kanal rename (T8-P8 özel durumu)
        # -------------------------
        rename_map = {}

        if 'T8-P8-1' in channel_names:
            rename_map['T8-P8-1'] = 'T8-P8'
        elif 'T8-P8-0' in channel_names and 'T8-P8' not in channel_names:
            rename_map['T8-P8-0'] = 'T8-P8'

        for old, new in rename_map.items():
            idx = channel_names.index(old)
            channel_names[idx] = new

        # -------------------------
        # 2️⃣ Kanal dictionary yap
        # -------------------------
        ch_dict = {
            ch: signals[i]
            for i, ch in enumerate(channel_names)
        }

        n_samples = signals.shape[1]

        # -------------------------
        # 3️⃣ Final kanal sırasını oluştur
        # -------------------------
        fixed_signals = []

        missing_channels = []

        for ch in final_channels:

            if ch in ch_dict:
                fixed_signals.append(ch_dict[ch])
            else:
                # Eksik kanal -> 0 doldur
                missing_channels.append(ch)
                fixed_signals.append(np.zeros(n_samples))

        if missing_channels:
            print(
                f"⚠️ {os.path.basename(file_path)} eksik kanallar: "
                f"{missing_channels} -> 0 ile dolduruldu"
            )

        fixed_signals = np.stack(fixed_signals, axis=0)

        return fixed_signals, final_channels, fs

    except Exception as e:
        print(f"❌ Kanal hatası ({os.path.basename(file_path)}): {e}")
        return None, None, None
