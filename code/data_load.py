import pandas as pd
import torch.utils.data as data
from PIL import Image
import numpy as np
from scipy.io import loadmat
from os import path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import random
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split

    
def load_audio(path):
    audio, sr = torchaudio.load(path)
    return audio, sr

def double_channel(sig):
    audio, sr = sig
    if audio.shape[0] == 2:
        return sig
    duplicated = torch.cat([audio, audio])
    return duplicated, sr

def downsample(sig, new_sr=22050):
    audio, sr = sig

    if sr == 22050:
        return sig
    first_channel = T.Resample(sr, new_sr)(audio[:1, :])
    second_channel = T.Resample(sr, new_sr)(audio[1:, :])
    res = torch.cat([first_channel, second_channel])
    # res = first_channel
    return res, new_sr

def append_trunc(sig, milis=3000):
    audio, sr = sig
    rows, audio_len = audio.shape
    max_len = sr // 1000 * milis

    if audio_len > max_len:
        audio = audio[:, :max_len]
    elif audio_len < max_len:
        diff = max_len - audio_len
        append_start_len = random.randint(0, diff)
        append_stop_len = diff - append_start_len
        append_start = torch.zeros((rows, append_start_len))
        append_stop = torch.zeros((rows, append_stop_len))

        audio = torch.cat((append_start, audio, append_stop), 1)
    return audio, sr

def spectro_mfcc(sig):
    audio, sr = sig
    mfcc_transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=64,
        melkwargs={"n_fft": 512, "n_mels": 64, "hop_length": None, "mel_scale": "htk"},
    )
    mfcc = mfcc_transform(audio)
    spec = T.AmplitudeToDB(top_db=80)(mfcc)
    return spec  # shape [channel, n_mels, time]


class SpeechDataset(data.Dataset): #data就是file_path
    def __init__(self, data, labels, index = False, twice = False):
        super(SpeechDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.index = index
        self.twice = twice

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        audio_file = self.data[idx]
        class_id = self.labels[idx]
        aud = load_audio(audio_file)
        rechannel = double_channel(aud)
        downsampl =downsample(rechannel)
        timed = append_trunc(downsampl)
        specgram = spectro_mfcc(timed)
        # print(specgram.shape)
        if self.twice == True:
            specgram = [specgram, specgram]

        if self.index == False:
            return specgram, class_id
        else:
            return specgram, class_id, idx

class DigitFiveDataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None, index = False, twice=False):
        super(DigitFiveDataset, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.twice = twice
        self.index=index

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        if img.shape[0] != 1:
            # transpose to Image type,so that the transform function can be used
            img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))

        elif img.shape[0] == 1:
            im = np.uint8(np.asarray(img))
            # turn the raw image into 3 channels
            im = np.vstack([im, im, im]).transpose((1, 2, 0))
            img = Image.fromarray(im)

        # do transform with PIL
        if self.transform is not None:
            if self.twice == False :
                img = self.transform(img)
            elif self.twice == True:
                img = [self.transform(img), self.transform(img)]
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.index == False and self.twice == False:
            return img, label.astype("int64")
        else:
            return img, label.astype("int64"), index
            

    def __len__(self):
        return self.data.shape[0]

    def append(self, b ,c, d):
        self.data = np.concatenate((self.data, b.data, c.data, d.data), axis = 0)
        self.labels = np.concatenate((self.labels, b.labels, c.labels, d.labels), axis = 0)
        return self


def load_gmu(index, twice):
    print("load gmu")
    df = pd.read_csv('code/short_df.csv')
    # speech_dataset = SpeechDataset(df, '', index)
    # num_items = len(speech_dataset)
    # num_train = round(num_items * 0.7)
    # num_test = num_items - num_train
    # train_ds, test_ds = random_split(speech_dataset, [num_train, num_test])
    # return train_ds, test_ds
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    train_ds = SpeechDataset(train_df['file_path'].to_numpy(), train_df['class_id'].to_numpy(), index, twice)
    test_ds = SpeechDataset(test_df['file_path'].to_numpy(),test_df['class_id'].to_numpy(), index, twice)
    return train_ds, test_ds

def load_mozilla(index, twice):
    print("load mozilla")
    df = pd.read_csv('code/short_df_mozilla_500.csv')
    # speech_dataset = SpeechDataset(df, '', index)
    # num_items = len(speech_dataset)
    # num_train = round(num_items * 0.7)
    # num_test = num_items - num_train
    # train_ds, test_ds = random_split(speech_dataset, [num_train, num_test])
    # return train_ds, test_ds
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    train_ds = SpeechDataset(train_df['file_path'].to_numpy(), train_df['class_id'].to_numpy(), index, twice)
    test_ds = SpeechDataset(test_df['file_path'].to_numpy(),test_df['class_id'].to_numpy(), index, twice)
    return train_ds, test_ds



def audio_dataset_read(domain, index=False, twice=False):
    if domain == "s":
        train_dataset, test_dataset = load_gmu(index, twice)
    elif domain == "m":
        train_dataset, test_dataset = load_mozilla(index, twice)
        print(len(train_dataset))
        print(len(test_dataset))

    else:
        raise NotImplementedError("Domain {} Not Implemented".format(domain))
    return train_dataset, test_dataset