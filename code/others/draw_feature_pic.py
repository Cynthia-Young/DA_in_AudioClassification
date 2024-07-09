import pandas as pd
import numpy as np
import librosa
import librosa.display
import torch
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import matplotlib.pyplot as plt


#忽略警告
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None

def normalize_audio_peak(audio, target_peak):
    # 计算音频的当前峰值
    current_peak = max(abs(audio))
    # 计算缩放系数
    scale = target_peak / current_peak
    # 对音频应用缩放系数
    normalized_audio = audio * scale
    return normalized_audio

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]

audio_file = './recordings/wav/english1.wav'
# audio_file = './cv-valid-train/sample-000044.wav'

sample_rate=32000
audio, sr = librosa.load(audio_file, sr = sample_rate, mono=True)

target_peak = 0.8
audio = normalize_audio_peak(audio, target_peak)
audio, index = librosa.effects.trim(audio, top_db=30)
clip_samples = sample_rate * 6
audio = pad_truncate_sequence(audio, clip_samples)

print(type(audio))
print(audio.shape)
audio = torch.from_numpy(audio).unsqueeze(0)
print(audio.shape)


spectrogram_extractor = Spectrogram(n_fft=1024, hop_length=320, win_length=1024, window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
logmel_extractor = LogmelFilterBank(sr=32000, n_fft=1024, n_mels=64, fmin=50, fmax=14000, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
x = spectrogram_extractor(audio)   # (batch_size, 1, time_steps, freq_bins)
logmel = logmel_extractor(x) 
final = spec_augmenter(logmel)

# 可视化对数梅尔语谱图特征
plt.figure(figsize=(10, 4))
plt.imshow(final.squeeze().detach().numpy(), aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Log Mel Spectrogram')
plt.xlabel('Time')
plt.ylabel('Mel Filter')
plt.tight_layout()
# 保存图像到文件
plt.savefig('logmel_spectrogram.png')
plt.show()