import os
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
import dill
import time
import random
import math
import json

import torch
import torchaudio
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB
import src.utils as utils

class AudioDataManager:
    def __init__(self, audio_folder) -> None:
        self.audio_folder = audio_folder
        self.train_songs = None
        self.valid_songs = None
        self.test_songs = None

    def create_song_splits(self, test_share, metadata_path="./meta.json", random_seed=123):
        unique_songs = list(set(int(s.split("_")[0]) for s in os.listdir(self.audio_folder)))
        random.Random(random_seed).shuffle(unique_songs)

        meta = {}
        idx = int(len(unique_songs)*(1-test_share))
        meta["train_songs"] = unique_songs[:idx]
        meta["test_songs"] = unique_songs[idx:]

        with open(metadata_path, "w") as fp:
            json.dump(meta, fp)

    def load_song_splits(self, metadata_path="./meta.json"):
        with open(metadata_path, "r") as fp:
            meta = json.load(fp)
        
        self.train_songs = meta["train_songs"]
        self.test_songs = meta["test_songs"]

    def make_valid_split(self, valid_share, random_seed=123):
        assert self.train_songs is not None, "Call load_song_splits first"

        songs = self.train_songs.copy()
        random.Random(random_seed).shuffle(songs)
        
        idx = int(len(songs)*(1-valid_share))
        self.train_songs = songs[:idx]
        self.valid_songs = songs[idx:]

# Generating Data for Processing Sequences of Multiple Spectorgrams ----------- 
def break_song_into_samples(x, max_seq_len=10):
    x.reset_index(inplace=True, drop=True)
    n_splits = math.ceil(x.shape[0] / max_seq_len)
    x["sample_group"] = 0
    for i in range(n_splits):
        x.loc[i*max_seq_len:max_seq_len*(i+1), "sample_group"] = int(i)
    
    x["song_clip_id"] = x["video_id"].astype(str) + "_" + x["sample_group"].astype(str) 
    return x

def _generate_seq_data_meta(metadata_path, max_seq_len=10):
    meta = pd.read_csv(metadata_path)
    meta['video_id'] = meta.audio_clip.str.split("_").apply(lambda l : l[0])
    sg = meta.groupby("video_id")
    song_groups = [sg.get_group(x) for x in sg.groups]
    song_groups = [break_song_into_samples(sg.copy(), max_seq_len) for sg in song_groups]
    return pd.concat(song_groups).reset_index(drop=True)

def generate_seq_data_arrays(metadata_path,
                            audio_folder = "./audio_wav",
                            save_folder  = "./data_arrays",
                            max_seq_length = 10,   #longest no. of audio clips to include per-sample
                            remake_existing = True,
                            #librosa:
                            sample_rate = 16000, ##to load audio as
                            n_mels      = 128,   ##number of Mel bands to generate
                            n_fft       = 2048,  ##length of the FFT window
                            hop_length  = 512,   ##number of samples between successive frames
                            verbosity   = 100):
    """Group clips (and their corresponding palettes) into sequences of consecutive clips. Save as numpy arrays. 
    The max number of consecutive clips to be included per sample is equivalent to "max_seq_length".
     
    generate_seq_data_arrays(
        metadata_path="./audio_color_mapping.csv",
        audio_folder="./audio_wav",
        save_folder="./data_arrays",
        max_seq_length=5,
        remake_existing=False,
        verbosity=20
    )
    """

    save_folder = f"{save_folder}_seq{max_seq_length}"
    os.makedirs(save_folder, exist_ok=True)

    meta = _generate_seq_data_meta(metadata_path, max_seq_length)

    t0 = time.time()
    samples = meta.groupby("song_clip_id")
    for i,grp in enumerate(samples.groups):
        save_path = Path(save_folder) / Path(grp)

        if save_path.exists() and not remake_existing:
            continue

        sample = samples.get_group(grp)
        seq_specs = []
        seq_pals  = []
        for j, row in sample.iterrows():
            wav_path  = Path(audio_folder) / Path(row["audio_clip"])

            # create mel-spectorgram
            y, sr = librosa.load(wav_path, sr=sample_rate)
            spec = librosa.feature.melspectrogram(y=y, sr=sr,
                                                n_mels=n_mels, 
                                                n_fft=n_fft,
                                                hop_length=hop_length)
            # convert to log scale (dB scale)
            log_spec = librosa.amplitude_to_db(spec, ref=1.)
            seq_specs.append( log_spec[np.newaxis, ...] ) #add Channel dimension

            # convert color palette to np array
            pal = []
            for k in row.keys():
                if k.startswith("rgb"):
                    col = [int(c) for c in row[k].split()]
                    pal.append(col)
            seq_pals.append( np.array(pal) )


        seq_specs = np.stack(seq_specs)
        seq_pals  = np.stack(seq_pals)

        data = (seq_specs, seq_pals)
        # save
        with open(save_path, 'wb') as f:
            dill.dump(data, f)
        
        # print progress
        if verbosity > 0:
            if i % verbosity == 0:
                print(f"[et={time.time()-t0 :.2f}s]",
                    f"Songs converted: {i+1} / {len(samples.groups)} ({(i+1)*100 / len(samples.groups) :.2f}%)")
    print(f"Final songs converted: {i+1}")

# Generating individual spectrogram arrays -------------
def generate_data_arrays(metadata_path,
                         audio_folder,
                         save_folder,
                         remake_existing = True,
                         #librosa:
                         sample_rate = 16000, ##to load audio as
                         n_mels      = 128,   ##number of Mel bands to generate
                         n_fft       = 2048,  ##length of the FFT window
                         hop_length  = 512,   ##number of samples between successive frames
                         verbosity   = 100):
    """Iterate through saved WAV files and their corresponding color palettes and save as tuples of np arrays.
    """

    os.makedirs(save_folder, exist_ok=True)
    meta = pd.read_csv(metadata_path)

    t0 = time.time()
    for i, row in meta.iterrows():
        wav_file = row["audio_clip"]
        save_path = Path(save_folder) / Path(wav_file.split(".")[0])
        if save_path.exists() and not remake_existing:
            continue
        wav_path = Path(audio_folder) / Path(wav_file)

        # create mel-spectorgram
        y, sr = librosa.load(wav_path, sr=sample_rate)
        spec = librosa.feature.melspectrogram(y=y, sr=sr,
                                            n_mels=n_mels, 
                                            n_fft=n_fft,
                                            hop_length=hop_length)
        # convert to log scale (dB scale)
        log_spec = librosa.amplitude_to_db(spec, ref=1.)
        log_spec = log_spec[np.newaxis, ...] #add Channel dim

        # convert color palette to np array
        pal = []
        for k in row.keys():
            if k.startswith("rgb"):
                rgb = [int(c) for c in row[k].split()]
                pal.append(rgb)
        pal = np.array(pal)


        data = (log_spec, pal)
        # save
        with open(save_path, 'wb') as f:
            dill.dump(data, f)
        
        # print progress
        if verbosity > 0:
            if i % verbosity == 0:
                print(f"[et={time.time()-t0 :.2f}s]",
                    f"Clips converted: {i+1} / {meta.shape[0]} ({(i+1)*100 / meta.shape[0] :.2f}%)")


    
# Splitting data samples -------------------
def split_samples_by_song(data_dir=None, 
                          audio_dir="./audio_wav", 
                          splitmeta_path="./meta.json",
                          valid_share=0.1,
                          test_share=None,
                          random_seed=149):
    """Split samples into train, test, and valid splits, grouped by song.
    (To avoid leakage this method ensures that all of the samples created from
    a given song are in only one of the 3 splits).
    If train-test song ID's do not already exist at 'splitmeta_path', they are
    created and saved. 'test_share' must be provided if this is the case.
    Otherwise, an existing train-test split (as created by 
    AudioDataManager.create_song_splits) is loaded from 'splitmeta_path'.

    For each split (train, valid, test), returns a list of the filenames which belong to the split.
    If data_dir is provided, the function assumes that the final data files are stored there.
    If not provided, it assumes that the audio_dir is the source of the final data files.
    """
    a = AudioDataManager(audio_dir)

    assert splitmeta_path.split(".")[-1] == "json"
    if not os.path.exists(splitmeta_path):
        assert test_share is not None, "Test-share is a required arg if creating new train-test splits."
        print("Creating new train-test split. Saving song-ids to:", splitmeta_path)

        a.create_song_splits(test_share=test_share, 
                             metadata_path=splitmeta_path,
                             random_seed=random_seed)

    a.load_song_splits(splitmeta_path)
    a.make_valid_split(valid_share, random_seed=random_seed)
    
    if data_dir is not None:
        data_files = os.listdir(data_dir)
    else:
        data_files = os.listdir(audio_dir)
    train_paths = [f for f in data_files if int(f.split("_")[0]) in a.train_songs]
    valid_paths = [f for f in data_files if int(f.split("_")[0]) in a.valid_songs]
    test_paths  = [f for f in data_files if int(f.split("_")[0]) in a.test_songs]

    return train_paths, valid_paths, test_paths


# Modify X or Y in the dataloader
class TransformX:
    def __init__(self, n_patches, pad_method, spec_aug, flatten_patches) -> None:
        self.n_patches = n_patches
        self.pad_method = pad_method
        self.spec_aug = spec_aug
        self.flatten_patches = flatten_patches
    def __call__(self, X):
        # normalize spectrogram to [-1, 1]
        if self.spec_aug:
            X /= 100.
        else:
            X /= torch.amax(torch.abs(X))
        return utils.create_patched_input(X, self.n_patches, self.pad_method, self.flatten_patches)

class TransformY:
    def __init__(self, n_colors) -> None:
        self.n_colors = n_colors
    def __call__(self, y):
        if self.n_colors==1:
            y = utils.pick_highest_luminance(y)
        return y / 255. #scale to [0, 1]


# Torch Datasets ------------------------------
class AudioToSpecDataset(torch.utils.data.Dataset):
    def __init__(self,
                 audio_paths_list:str,
                 audio_data_dir:str,
                 audio_color_map:dict=None,
                 X_transform=None,
                 y_transform=None,
                 spec_augment=None,
                 # mel-spec params: default -> shape(1,128,160) mel-spec
                 n_fft:int=2048,
                 n_mels:int=128,
                 resample_freq:int=16000,
                 hop_length:int=501,
                 mono_wav:bool=True
                 ) -> None:
        """PyTorch Dataset which loads .WAV clips and prepares them as MelSpectrograms.

        spec_aug = torch.nn.Sequential(
            TimeStretch(stretch_factor, fixed_rate=True),
            FrequencyMasking(freq_mask_param=80),
            TimeMasking(time_mask_param=80),
        )
        """
        super().__init__()
        self.paths_list = audio_paths_list
        self.data_dir = Path(audio_data_dir)
        self.audio_color_map = audio_color_map

        self.X_transform = X_transform
        self.y_transform = y_transform
        
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.resample_freq = resample_freq
        self.mono = mono_wav
        self.hop_length = hop_length
        if self.hop_length is None:
            self.hop_length = n_fft // 2

        self.spec = Spectrogram(n_fft=n_fft, power=2, hop_length=self.hop_length)
        self.spec_aug = spec_augment
        self.mel_scale = MelScale(n_mels=self.n_mels,
                                  sample_rate=self.resample_freq, 
                                  n_stft=self.n_fft // 2 + 1)
        self.amp_to_db = AmplitudeToDB()

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, idx):
        wav_file  = self.paths_list[idx]
        audiopath = self.data_dir / Path(wav_file)

        # get X
        wav, sr = torchaudio.load(audiopath, normalize=True)
        if self.mono:
            wav = torch.mean(wav, dim=0).unsqueeze(0)
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.resample_freq)
        spec = self.spec(wav)
        X = self.amp_to_db(self.mel_scale(spec))
        if self.spec_aug:
            spec = self.spec_aug(spec)
        X_aug = self.amp_to_db(self.mel_scale(spec))

        # get Y
        y = None
        if self.audio_color_map is not None:
            y = self.audio_color_map[wav_file]
            y = torch.tensor(np.array(y, dtype=np.float32))

        X = self.preprocess_spec(X)
        X_aug = self.preprocess_spec(X_aug)
        y = self.preprocess_pal(y)
        return X, X_aug, y

    @staticmethod
    def gen_audio_color_map(metadata_path):
        ac_map = {}
        meta = pd.read_csv(metadata_path)
        for i, row in meta.iterrows():
            wav_file = row["audio_clip"]
            pal = []
            for k in row.keys():
                if k.startswith("rgb"):
                    rgb = [int(c) for c in row[k].split()]
                    pal.append(rgb)

            ac_map[wav_file] = pal
        return ac_map

    def preprocess_spec(self, X):
        if self.X_transform is not None:
            X = self.X_transform(X)
        return X

    def preprocess_pal(self, y):
        if self.y_transform is not None:
            y = self.y_transform(y)
        return y



class SimpleSpecDataset(torch.utils.data.Dataset):
    def __init__(self,
                 paths_list,
                 data_dir,
                 X_transform=None,
                 y_transform=None
                 ) -> None:
        super().__init__()
        self.paths_list = paths_list
        self.data_dir = Path(data_dir)
        self.X_transform = X_transform
        self.y_transform = y_transform

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, idx):
        p = self.data_dir / Path(self.paths_list[idx])

        with open(p, "rb") as f:
            X, y = dill.load(f)

        X = torch.tensor(self.preprocess_spec(X))
        y = torch.tensor(self.preprocess_pal(y))
        return X, y

    def preprocess_spec(self, X):
        # normalize spectrogram to [-1, 1]
        X /= np.amax(np.abs(X))
        if self.X_transform is not None:
            X = self.X_transform(X)
        return X

    def preprocess_pal(self, y):
        # scale all RGBs to [0,1]
        y = y.astype(np.float32) / 255.0
        if self.y_transform is not None:
            y = self.y_transform(y)
        return y




class SeqAudioRgbDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 paths_list:list,
                 data_dir:str, 
                 max_seq_length:int,
                 pad_short_seqs=True,
                 X_transform=None, 
                 y_transform=None,
                 *kwargs
                 ) -> None:
        super().__init__()
        self.data_dir    = Path(data_dir)
        self.paths_list  = paths_list
        self.X_transform = X_transform
        self.y_transform = y_transform
        self.pad_short_seqs = pad_short_seqs
        self.max_seq_len    = max_seq_length 
    
    def remove_short_seqs(self):
        """What to do with sequences consisting of < max_seq_len clips?
        Could try to pad them, and then deal with ignoring them in backprop,
        or for now, we can just remove them.
        """
        final_paths = [] 
        for i, p in enumerate(self.paths_list):
            with open(self.data_dir / Path(p), "rb") as f:
                X, y = dill.load(f)
            if X.shape[0] == self.max_seq_len:
                final_paths.append(p)
        self.paths_list = final_paths
        

    def _check_for_short_seqs(self):
        short = 0
        for p in self.paths_list:
            with open(self.data_dir / Path(p), "rb") as f:
                X,_ = dill.load(f)
            if X.shape[0] != self.max_seq_len:
                short += 1
        return short
        
    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, idx):
        p = self.data_dir / Path(self.paths_list[idx])

        with open(p, "rb") as f:
            X, y = dill.load(f)

        X = torch.tensor(self.preprocess_spec(X))
        y = torch.tensor(self.preprocess_pal(y))

        if X.shape[0] < self.max_seq_len and self.pad_short_seqs:
            X = self.pad_X(X)
            y = self.pad_y(y)

        return X, y

    def preprocess_spec(self, X):

        if self.X_transform is not None:
            X = self.X_transform(X)

        # normalize each spectrogram in the sequence to [-1, 1]
        seq_maxs = np.amax(np.abs(X), axis=(1,2,3)) #maximum along (C, H, W)
        seq_maxs = seq_maxs.reshape(seq_maxs.shape[0], 1, 1, 1) #(seq_len, C, H, W)
        X /= seq_maxs
        # for i in range(X.shape[0]): ##seq_length
        #     X[i, :] = X[i,:] / np.max(np.abs(X[i,:]))
        return X

    def preprocess_pal(self, y):
        # scale all RGBs to [0,1]
        if self.y_transform is not None:
            y = self.y_transform(y)
        y = y.astype(np.float32) / 255.
        return y
    
    def pad_X(self, X):
        pad_arr = torch.empty((1, *X.shape[1:]))
        diff = self.max_seq_len - X.shape[0]
        pads = [torch.zeros_like(pad_arr) for _ in range(diff)]
        X = torch.cat(pads + [X])

        assert X.shape[0] == self.max_seq_len
        return X
    
    def pad_y(self, y):
        pad_arr = torch.empty((1, y.shape[1], 3)) #seq_len, n_colors, rgb[3]
        diff = self.max_seq_len-y.shape[0]
        pads = [torch.zeros_like(pad_arr) for _ in range(diff)]
        y = torch.cat(pads + [y])

        assert y.shape[0] == self.max_seq_len
        return y




# if __name__=="__main__":

#     generate_seq_data_arrays(
#         metadata_path="./data/pop_videos/audio_color_mapping.csv",
#         audio_folder="./data/pop_videos/audio_wav",
#         save_folder="./data_arrays",
#         max_seq_length=5,
#         remake_existing=False,
#         verbosity=100
#     )