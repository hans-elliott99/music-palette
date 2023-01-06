import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import librosa
from pathlib import Path
import dill
import time
from sklearn.model_selection import train_test_split
import random
import math


# Load dataset
def load_dataset(data_folder="./audio_mel", norm_spect=True, norm_pal_rgbs=True, n_samples=None):
    clip_list = os.listdir(data_folder)

    # ensure clips are sorted by song-clip
    clip_list = [(int(c.split("_")[0]), int(c.split("_")[-1]), c) for c in clip_list]
    clip_list.sort(reverse=False)

    song_ids, spectrograms, palettes = [], [], []
    count = 0
    for song_id, clip_id, clip in clip_list:
        with open(Path(data_folder) / Path(clip), 'rb') as f:
            s, p = dill.load(f)  
        if norm_spect:
            s = s.astype(np.float32) / s.max()
        if norm_pal_rgbs:
            p = p.astype(np.float32) / 255
        song_ids.append(song_id)
        spectrograms.append(s[np.newaxis, ...].astype(np.float32)) ##add "channel" dim
        palettes.append(p.astype(np.float32))
        count += 1
        if n_samples is not None and count >= n_samples:
            break
    
    return spectrograms, palettes, song_ids
    
def get_inds(ls, song_id:int):
    return [i for i,el in enumerate(ls) if el==int(song_id)]

def load_and_split_data(data_folder,
                        norm_spectogram=True,
                        norm_palette_rgbs=True,
                        n_samples=None,
                        test_size=0.2,
                        valid_size=0.1,
                        random_seed=123):
    # load and make train, valid, test splits, stratified by song
    spec, pal, song_ids = load_dataset(data_folder, norm_spectogram, norm_palette_rgbs, n_samples)
    unique_songs = list(set(song_ids))
    random.Random(random_seed).shuffle(unique_songs)

    nontest     = unique_songs[:int( len(unique_songs)*(1-test_size ))]
    test_songs  = unique_songs[int( len(unique_songs)*(1-test_size) ):]
    train_songs = nontest[:int( len(nontest)*(1-valid_size) )]
    valid_songs = nontest[int( len(nontest)*(1-valid_size) ):]

    X_train, X_val, X_test = [], [], []
    Y_train, Y_val, Y_test = [], [], []
    s_train, s_val, s_test = [], [], []

    for song in train_songs:
        inds = get_inds(song_ids, song)
        X_train += [spec[i] for i in inds]
        Y_train += [pal[i]  for i in inds]
        s_train += [song_ids[i] for i in inds]
        
    for song in valid_songs:
        inds = get_inds(song_ids, song)
        X_val += [spec[i] for i in inds]
        Y_val += [pal[i] for i in inds]
        s_val += [song_ids[i] for i in inds]

    for song in test_songs:
        inds = get_inds(song_ids, song)
        X_test += [spec[i] for i in inds]
        Y_test += [pal[i] for i in inds]
        s_test += [song_ids[i] for i in inds]

    return ((X_train, X_val, X_test), (Y_train, Y_val, Y_test), (s_train, s_val, s_test))

# prep next song (split into multiple samples if too long)
def prep_next_song(song_ids_split, ##song ids 
                   spect_split,    ##spectorgrams
                   pal_split,      ##palettes
                   song_id,        ##current song id (int)
                   max_clips_per_song=20, ##longest possible sample is max_clips_per_song + max_clips_grace - 1
                   max_clips_grace=0):    ##if last sample would be less than this many clips, tack onto the previous sample
    """Given the next song (via song_id), split song into individual samples.
    (Note, this is one sample, we need to make a batch) 
    """

    # break song into samples
    final_song_inds = []
    final_spects    = []
    final_pals      = []
    inds = get_inds(song_ids_split, int(song_id))
    if len(inds) > max_clips_per_song:
        n_splits = math.ceil(len(inds)/max_clips_per_song)

        splt_cnt = 0
        while True:
            start_idx = splt_cnt * max_clips_per_song
            end_idx   = (splt_cnt+1) * max_clips_per_song
            if (len(inds) - end_idx) < max_clips_grace:
                end_idx = len(inds)
                splt_cnt += 1

            final_song_inds.append( song_ids_split[start_idx:end_idx] )
            final_spects.append( np.array(spect_split[start_idx:end_idx]) )
            final_pals.append( np.array(pal_split[start_idx:end_idx]) )

            splt_cnt += 1
            if splt_cnt >= n_splits:
                break

    return final_song_inds, final_spects, final_pals


@torch.no_grad()
def rgb_accuracy(pred_tensor, truth_tensor, window_size, scale_rgb=True):
    assert pred_tensor.size(0) == truth_tensor.size(0)
    pred = pred_tensor.clone()
    truth = truth_tensor.clone()

    if scale_rgb:
        pred *= 255
        truth *= 255
    window = int(window_size/2)

    batch_sum = 0
    for b in range(pred.size(0)):
        correct = 0
        for pred_color, true_color in zip(pred[b], truth[b]):
            pred_color = pred_color.tolist()
            true_color = true_color.tolist()
            for v in range(3):
                if true_color[v]-window <= pred_color[v] <= true_color[v]+window:
                    correct += 1
        batch_sum += correct
    
    return batch_sum / (pred.size(0) * pred.size(1) * 3) #batch_size * n_colors * 3[rgb]


def quick_color_display(rgb:list, h=30, w=30):
    """plt.imshow(quick_color_display([10, 123, 30]))
    """
    rgb = [round(v) for v in rgb]
    block = np.zeros((h, w, 3))
    block[:,:,:] = rgb
    return block.astype(np.uint8)

#import colorsys
#colorsys.rgb_to_hsv
# def rgb_to_hue(rgb:np.array):
#     rgb = rgb.astype(np.float64) / 255.
#     max_i = np.argmax(rgb)
#     min_i = np.argmin(rgb)
#     diff  = max_i - min_i
#     if diff == 0:
#         hue = 0
#     else:
#         match max_i:
#             case 0: #r
#                 hue = (rgb[1]-rgb[2]) / diff #g-b / max-min
#             case 1: #g
#                 hue = 2. + (rgb[2]-rgb[0]) / diff #2 + b-r / max-min
#             case 2: #b
#                 hue = 4. + (rgb[0]-rgb[1]) / diff #5 + r-g / max-min
#         hue *= 60
#         if hue < 0:
#             hue += 360
#     return hue


def generate_spectograms(meta,
                         audio_folder = "./audio_wav", 
                         save_folder  = "./audio_mel",
                         remake_existing = False,
                         #librosa
                         sample_rate = 16000, ##to load audio as
                         n_mels      = 128,   ##number of Mel bands to generate
                         n_fft       = 2048,  ##length of the FFT window
                         hop_length  = 512,   ##number of samples between successive frames
                         verbosity   = 100
                         ):
    os.makedirs(save_folder, exist_ok=True)

    # iterate through all clips - create mel-spectogram and save data
    t0 = time.time()
    for i, row in meta.iterrows():
        clip = row["audio_clip"]
        wav_path  = Path(audio_folder) / Path(clip)
        save_path = Path(save_folder) / Path(clip.split(".")[0])

        # create mel-spectorgram
        if save_path.exists() and not remake_existing:
            continue
        
        y, sr = librosa.load(wav_path, sr=sample_rate)
        spec = librosa.feature.melspectrogram(y=y, sr=sr,
                                             n_mels=n_mels, 
                                             n_fft=n_fft,
                                             hop_length=hop_length)
        # convert to log scale (dB scale)
        log_spec = librosa.amplitude_to_db(spec, ref=1.)

        # convert color palette to np array
        pal = []
        for k in row.keys():
            if k.startswith("rgb"):
                col = [int(c) for c in row[k].split()]
                pal.append(col)
        pal = np.array(pal)

        data = (log_spec, pal) ##X, y
        # save
        with open(save_path, 'wb') as f:
            dill.dump(data, f)
        
        # print progress
        if verbosity > 0:
            if i % verbosity == 0:
                print(f"[et={time.time()-t0 :.2f}s]",
                    f"Data converted: {i+1} / {meta.shape[0]} ({(i+1)*100 / meta.shape[0] :.2f}%)")