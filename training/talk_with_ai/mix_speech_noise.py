import numpy as np
import librosa
from scipy.io import wavfile
def mix_audio(clean_wav_path, noise_wav_path, output_path, noise_snr=0):
    sr, clean_wav = wavfile.read(clean_wav_path)
    noise_wav, _ = librosa.load(noise_wav_path)
    
    clean_wav = clean_wav.astype(np.float32)
    noise_wav = noise_wav.astype(np.float32)
    
    if type(noise_snr) == int or type(noise_snr) == float:
        snr = noise_snr
    elif type(noise_snr) == tuple:
        snr = np.random.randint(noise_snr[0], noise_snr[1]+1)
    
    clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
    if len(clean_wav) > len(noise_wav):
        ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
        noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
    if len(clean_wav) < len(noise_wav):
        start = 0
        noise_wav = noise_wav[start: start + len(clean_wav)]
    
    noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
    adjusted_noise_rms = clean_rms / (10**(snr/20))
    adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
    
    # Giảm âm lượng noise xuống
    # noise_reduction_factor = 0.2  # Giảm âm lượng noise xuống 20%
    # adjusted_noise_wav = adjusted_noise_wav * noise_reduction_factor
    
    mixed = clean_wav + adjusted_noise_wav

    # Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
        if mixed.max(axis=0) >= abs(mixed.min(axis=0)): 
            reduction_rate = max_int16 / mixed.max(axis=0)
        else:
            reduction_rate = min_int16 / mixed.min(axis=0)
        mixed = mixed * (reduction_rate)
    
    mixed = mixed.astype(np.int16)
    wavfile.write(output_path, sr, mixed)

# mix_audio('/data/datdt/mvlrs_v1/audio/short-pretrain/5535415699068794046/00002.wav', 'Noise/speech/mixed_speech_18.wav', 'test.wav')

#Đọc file test.tsv, mix ngẫu nhiên 1 noise ở trong folder Noise vào từng audio_path trong file test.tsv
import pandas as pd
import os
# Đọc file test.tsv
df = pd.read_csv('/data/datdt/mvlrs_v1/train2.tsv', sep='\t')
import random
noise_files = [os.path.join(root, file) for root, _, files in os.walk('vi_noise/cut') for file in files if file.endswith('.mp3')]
print(len(noise_files))
# Lặp qua từng dòng trong file test.tsv
for index, row in df.iterrows():    
    print(row['path'])
    # Lấy audio_path và noise_path từ row
    audio_path = "/data/datdt/mvlrs_v1/audio/" + row['path']
    sentence = row['sentence'].lower()
    subfolder = row['path'].split('/')[1]
    audio_name = row['path'].split('/')[2]
    # Tạo subfolder nếu chưa có
    if not os.path.exists(f"/data/datdt/mvlrs_v1/train2noise_vi2/{subfolder}"):
        os.makedirs(f"/data/datdt/mvlrs_v1/train2noise_vi2/{subfolder}") 
    noise_path = random.choice(noise_files)
    mix_audio_path = f"/data/datdt/mvlrs_v1/train2noise_vi2/{subfolder}/{audio_name}"
    # Mix audio và noise    
    mix_audio(audio_path, noise_path, mix_audio_path)
    # Lưu path và sentence vào file test_noise.tsv
    with open('/data/datdt/mvlrs_v1/train2noise_vi2.tsv', 'a') as f:
        f.write(f"train2noise_vi2/{subfolder}/{audio_name}\t{sentence}\n")
