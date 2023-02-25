
from src.utils import quick_color_display, create_patched_input, plot_predicted_melspec
from src.models.transformers import GPTPatchTransformer

import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import pyaudio
import wave


device_index = 0
model_path = "./checkpoints/gtzan_ft_0.pth"
task = "genre" #color, genre, or spectrogram

temp_recording = "_temp_.wav"
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5 ##model-dependent
GTZAN_CLASSES = {"blues":0, "classical":1, "country":2, "disco":3, "hiphop":4,
                  "jazz":5, "metal":6, "pop":7, "reggae":8, "rock":9}
i2tgzan = {int(v):k for k,v in GTZAN_CLASSES.items()}

def record_next_clip(pyaudio_stream:pyaudio.Stream, sample_width, wav_filename):
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = pyaudio_stream.read(CHUNK)
        frames.append(data)

    with wave.open(wav_filename, "wb") as wavf:
        wavf.setnchannels(CHANNELS)
        wavf.setsampwidth(sample_width)
        wavf.setframerate(RATE)
        wavf.writeframes(b''.join(frames))

    return frames

def convert_to_melspec_tensor(wav_filename,
                            #librosa:
                            sample_rate = 16000, ##to load audio as
                            n_mels      = 128,   ##number of Mel bands to generate
                            n_fft       = 2048,  ##length of the FFT window
                            hop_length  = 512,   ##number of samples between successive frames
                            ):
    y, sr = librosa.load(wav_filename, sr=sample_rate)
    spec = librosa.feature.melspectrogram(y=y, sr=sr,
                                          n_mels=n_mels, 
                                          n_fft=n_fft,
                                          hop_length=hop_length)
    # convert to log scale (dB scale)
    log_spec = librosa.amplitude_to_db(spec, ref=1.)
    #add channel dimension. shape=(Channels[1], Height[features/n_mels], Width[time])
    log_spec = log_spec[np.newaxis, ...] 
    return log_spec

def preprocess_spec(log_spec_array):
    # normalize to [-1, 1]
    log_spec_array /= np.max(np.abs(log_spec_array)) 
    return torch.tensor(log_spec_array).unsqueeze(0) # add batch and time dimensions

def get_gpt_input(X):
    # Add "start" patch to top of sequence (-1 tensor of shape (1,1,feats))...
    # Remove last patch in sequence...
    # Since model is using one patch to predict the next.
    batch_size, _, feats = X.shape
    start_patch = (torch.zeros(batch_size, 1, feats) - 1).to(X.device) #(b, 1, feats)
    return torch.column_stack( (start_patch, X[:, :-1, :]) )           #(b, n_patches, feats)

def plot_palette(logits_flat, n_colors):
    pal_arr = logits_flat.unsqueeze(0).reshape(n_colors,3) #remove batch dim
    
    fig = plt.figure()
    fig.suptitle("Predicted Palette")
    for i in range(n_colors):
        color = [int(c*255) for c in pal_arr[i].tolist()]
        ax = plt.subplot(n_colors, 1, i+1) ##subplot inds start at 1
        ax.set_title("RGB: " + ', '.join([str(c) for c in color]))
        ax.imshow(quick_color_display(color))

    plt.show()

def plot_palette_with_melspec(logits_flat, n_colors, melspec=None):
    pal_arr = logits_flat.unsqueeze(0).reshape(n_colors,3)
    strip = []
    for i in range(n_colors):
        color = [int(c*255) for c in pal_arr[i].tolist()]
        strip.append(quick_color_display(color))
    strip = np.vstack(strip)

    plt.figure()
    if melspec is not None:
        plt.subplot(121)
        plt.imshow(melspec)
        pal_pos = 122
    else:
        pal_pos = 111
    plt.subplot(pal_pos)
    plt.imshow(strip)
    # plt.show()






def main():
    device = torch.device("cpu") #if torch.cuda.is_available() else torch.device("cpu")
    
    # Load in model
    print("loading model")

    model = torch.load(model_path, map_location=device)
    model.eval()

    # Record audio
    p = pyaudio.PyAudio()

    spec_sequence = None
    for i in range(10): #while loop
        stream = p.open(rate=RATE, 
                        channels=CHANNELS,
                        format=FORMAT,
                        input=True,
                        input_device_index=device_index)

        print("recording...")
        record_next_clip(stream, p.get_sample_size(FORMAT), temp_recording)

        stream.stop_stream()
        stream.close()

        # Convert audio to mel-spec
        spec_t = convert_to_melspec_tensor(temp_recording)
        spec_t = preprocess_spec(spec_t) #shape (batch[1], C, H, W)
        X = create_patched_input(spec_t, n_patches=20, pad_method="min")

        if task.lower().startswith("spec"):
            X_gpt = get_gpt_input(X)
            with torch.no_grad():
                logits = model(X_gpt, pretraining=True)   #(1, n_feats)
    
            plot_predicted_melspec(spec_t[0].cpu(), logits[0].cpu(), figsize=(10,8))

        elif task.lower().startswith("color"): #generative pretraining
            with torch.no_grad():
                logits = model(X)
            # Do something with the color palette
            ## Let's plot the most recent mel and the most recently generated palette
            mel = spec_t.reshape((128, 157, 1))
            plot_palette_with_melspec(logits, int(model.n_classes/3), mel)
            plt.show()
        
        elif task.lower().startswith("genre"):
            with torch.no_grad():
                logits = model(X)
            pred = logits.max(1).indices
            print(f"Predicted Genre: {i2tgzan[pred.item()]}")

    p.terminate()
    os.remove(temp_recording)

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt: Exiting gracefully.")
        os.remove(temp_recording)

    
