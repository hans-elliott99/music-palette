
from src.utils import quick_color_display, create_patched_input, plot_predicted_melspec
from src.models.transformers import ConvTransformer, PatchTransformer, GPTPatchTransformer

import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt

import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5 ##model-dependent
device_index = 0
model_path = "./checkpoints/gpt_base.pth"
temp_recording = "_temp.wav"


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
    return torch.tensor(log_spec_array).unsqueeze(0).unsqueeze(0) # add batch and time dimensions


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

if __name__=="__main__":
    device = torch.device("cpu") #if torch.cuda.is_available() else torch.device("cpu")
    
    # Load in model
    print("loading model")

    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    if isinstance(model, (ConvTransformer)):
        mod = "conv"
        block_size = model.max_seq_len
    elif isinstance(model, (PatchTransformer, GPTPatchTransformer)):
        mod = "patch"
        n_patches = model.n_patches
    else: 
        raise NotImplementedError("Model type unrecognized.") 


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
        spec_t = preprocess_spec(spec_t) #shape (batch[1], time[1], C, H, W)

        if mod == "conv":
            if spec_sequence is None:
                spec_sequence = spec_t
            else:
                # Keep last (block_size-1) specs and append the new one by stacking
                # on the time dimension (axis=1 or "column stack")
                spec_sequence = torch.column_stack((
                    spec_sequence[:, -(block_size-1):, :], 
                    spec_t
                )) 
            
            spec_sequence.to(device)
            print(spec_sequence.shape)
            # Send through model 
            with torch.no_grad():
                logits_flat = model(spec_sequence) #(1, seq_len, n_colors*3)

        elif mod == "patch":
            with torch.no_grad():
                x = create_patched_input(spec_t.squeeze(1), n_patches=20, pad_method="min")
                print(x.shape)                    #(1, n_patches, feats)
                logits_flat = model(x, pretraining=True)            #(1, n_colors*3)
                logits_flat = logits_flat.unsqueeze(1) #(1, 1, n_colors*3) for consistency w/ conv

        if True: #generative pretraining
            plot_predicted_melspec(spec_t[0][0].cpu(), logits_flat[0].cpu(), figsize=(12,10))

        # Do something with the color palette
        ## Let's plot the most recent mel and the most recently generated palette
        if False:
            mel = spec_t.reshape((128, 157, 1))
            print(logits_flat[0, -1, :] * 255)
            plot_palette_with_melspec(logits_flat[:, -1, :], model.n_colors, mel)
            plt.show()


    p.terminate()
 

# when to resume recording again? immediately after inference or after some time has passed?
# how often to reset the hidden state?




    
