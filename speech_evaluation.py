import os
import numpy as np
np.complex = complex  # Temporary fix for librosa
import pandas as pd
import librosa
import soundfile as sf
from pesq import pesq
from pystoi.stoi import stoi
from scipy.stats import pearsonr
from tqdm import tqdm

# Constants
SAMPLE_RATE = 16000
REAL_ROOT = './Processed_Real'
FAKE_ROOT = './Processed_Fake'
METHODS = ['GAN', 'LLM', 'TTS', 'VC']
CSV_FILENAME = 'objective_measures.csv'

# Helper: Compute MCD
def compute_mcd(x, y, n_mfcc=13):
    x_mfcc = librosa.feature.mfcc(y=x, sr=SAMPLE_RATE, n_mfcc=n_mfcc)[1:]
    y_mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=n_mfcc)[1:]
    min_len = min(x_mfcc.shape[1], y_mfcc.shape[1])
    x_mfcc, y_mfcc = x_mfcc[:, :min_len], y_mfcc[:, :min_len]
    diff = x_mfcc - y_mfcc
    mcd = np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))
    mcd *= (10.0 / np.log(10)) * np.sqrt(2)
    return mcd

# Process all files and collect metrics
data_rows = []

for method in METHODS:
    real_path = os.path.join(REAL_ROOT, method)
    fake_path = os.path.join(FAKE_ROOT, method)

    real_files = sorted([f for f in os.listdir(real_path) if f.endswith('.wav')])
    fake_files = sorted([f for f in os.listdir(fake_path) if f.endswith('.wav')])
    common_files = set(real_files) & set(fake_files)

    for file in tqdm(common_files, desc=f'Processing {method}'):
        real_file = os.path.join(real_path, file)
        fake_file = os.path.join(fake_path, file)

        # Load and resample
        real, _ = librosa.load(real_file, sr=SAMPLE_RATE)
        fake, _ = librosa.load(fake_file, sr=SAMPLE_RATE)

        min_len = min(len(real), len(fake))
        real, fake = real[:min_len], fake[:min_len]

        try:
            pesq_score = pesq(SAMPLE_RATE, real, fake, 'nb')
        except Exception:
            pesq_score = np.nan

        msd_score = np.mean((real - fake) ** 2)

        try:
            pcc_score, _ = pearsonr(real, fake)
        except Exception:
            pcc_score = np.nan

        try:
            stoi_score = stoi(real, fake, SAMPLE_RATE, extended=False)
        except Exception:
            stoi_score = np.nan

        try:
            mcd_score = compute_mcd(real, fake)
        except Exception:
            mcd_score = np.nan

        data_rows.append([
            real_file,
            fake_file,
            pesq_score,
            msd_score,
            pcc_score,
            stoi_score,
            mcd_score
        ])

# Save to CSV
columns = ['Real_File', 'Fake_File', 'PESQ', 'MSD', 'PCC', 'STOI', 'MCD']
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv(CSV_FILENAME, index=False)

print(f"✅ CSV file saved: {CSV_FILENAME}")

