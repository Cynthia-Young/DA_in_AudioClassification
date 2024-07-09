import pandas as pd
import torch.utils.data as data
import numpy as np
from sklearn.model_selection import train_test_split
import librosa

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]

def normalize_audio_peak(audio, target_peak):
    # 计算音频的当前峰值
    current_peak = max(abs(audio))
    # 计算缩放系数
    scale = target_peak / current_peak
    # 对音频应用缩放系数
    normalized_audio = audio * scale
    return normalized_audio
    
def feature(audio_path, sample_rate=32000):
    (audio, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    target_peak = 0.8
    audio = normalize_audio_peak(audio, target_peak)
    audio, index = librosa.effects.trim(audio, top_db=30) # 去除静音段
    clip_samples = sample_rate * 6
    audio = pad_truncate_sequence(audio, clip_samples)
    return audio.astype(np.float32)

def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target

class SpeechDataset(data.Dataset): #data就是file_path
    def __init__(self, data, labels, classes_num, index = False, twice = False):
        super(SpeechDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.index = index
        self.twice = twice
        self.classes_num = classes_num

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        audio_file = self.data[idx]
        aud = feature(audio_file)

        class_id = self.labels[idx]
        class_id = to_one_hot(class_id, self.classes_num)
        
        if self.twice == True:
            aud = [aud, aud]

        if self.index == False:
            return aud, class_id
        else:
            return aud, class_id, idx


def load_gmu(args, index, twice):
    print("load gmu")
    df = pd.read_csv('code/short_df.csv')
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    train_ds = SpeechDataset(train_df['file_path'].to_numpy(), train_df['class_id'].to_numpy(), args.class_num, index, twice)
    test_ds = SpeechDataset(test_df['file_path'].to_numpy(),test_df['class_id'].to_numpy(), args.class_num, index, twice)
    return train_ds, test_ds

def load_mozilla(args, index, twice):
    print("load mozilla")
    df = pd.read_csv('code/short_df_mozilla_500.csv')
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    train_ds = SpeechDataset(train_df['file_path'].to_numpy(), train_df['class_id'].to_numpy(), args.class_num, index, twice)
    test_ds = SpeechDataset(test_df['file_path'].to_numpy(),test_df['class_id'].to_numpy(), args.class_num, index, twice)
    return train_ds, test_ds


def audio_dataset_read(args, domain, index=False, twice=False):
    if domain == "s":
        train_dataset, test_dataset = load_gmu(args, index, twice)
    elif domain == "m":
        train_dataset, test_dataset = load_mozilla(args, index, twice)
        print(len(train_dataset))
        print(len(test_dataset))

    else:
        raise NotImplementedError("Domain {} Not Implemented".format(domain))
    return train_dataset, test_dataset