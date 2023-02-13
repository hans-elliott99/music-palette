

import torch
from models.transformers import ConvTransformer

import utils
import numpy as np

import pyaudio
import wave
import librosa

import matplotlib.pyplot as plt

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5 ##model-dependent
device_index = 0
model_path = "./checkpoints/transformer_1.pth"


@torch.no_grad()
def rnn_forwardpass_spectrogram(model, spec_array, hidden):
    logits_flat, hidden = model(spec_array, hidden)
    return logits_flat, hidden

@torch.no_grad()
def forwardpass_spectrogram(model, spec_array, device):
    return model(spec_array)


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
                            verbosity   = 100):

    y, sr = librosa.load(wav_filename, sr=sample_rate)
    spec = librosa.feature.melspectrogram(y=y, sr=sr,
                                          n_mels=n_mels, 
                                          n_fft=n_fft,
                                          hop_length=hop_length)
    # convert to log scale (dB scale)
    log_spec = librosa.amplitude_to_db(spec, ref=1.)
    # normalize to [-1, 1]
    log_spec /= np.max(np.abs(log_spec)) 
    log_spec = log_spec[np.newaxis, ...] #add Channel dim

    return torch.tensor(log_spec).unsqueeze(0) ##add Batch dim


def plot_palette(logits_flat, n_colors):
    pal_arr = logits_flat.unsqueeze(0).reshape(n_colors,3) #remove batch dim
    
    fig = plt.figure()
    fig.suptitle("Predicted Palette")
    for i in range(n_colors):
        color = [int(c*255) for c in pal_arr[i].tolist()]
        ax = plt.subplot(n_colors, 1, i+1) ##subplot inds start at 1
        ax.set_title("RGB: " + ', '.join([str(c) for c in color]))
        ax.imshow(utils.quick_color_display(color))

    plt.show()

def plot_palette_strip(logits_flat, n_colors, melspec=None):
    pal_arr = logits_flat.unsqueeze(0).reshape(n_colors,3)
    strip = []
    for i in range(n_colors):
        color = [int(c*255) for c in pal_arr[i].tolist()]
        strip.append(utils.quick_color_display(color))
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

    model:ConvTransformer = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    block_size = model.max_seq_len

    # Record audio
    p = pyaudio.PyAudio()

    specs = None
    for i in range(10): #while loop
        stream = p.open(rate=RATE, 
                        channels=CHANNELS,
                        format=FORMAT,
                        input=True,
                        input_device_index=device_index)

        print("recording...")
        record_next_clip(stream, p.get_sample_size(FORMAT), "test.wav")

        stream.stop_stream()
        stream.close()

        # Convert audio to mel-spec
        spec_t = convert_to_melspec_tensor("test.wav")
        spec_t = spec_t.unsqueeze(0) #batch dim

        if specs is None:
            specs = spec_t
        else:
            specs = torch.column_stack((specs[:, -(block_size-1):, :], spec_t)) #keep last (block_size-1) specs and append the new one (on Time dim, 1)
        specs.to(device)
        
        print(specs.shape)
        # Send through model 
        # hidden = model.init_hidden(batch_size=1).to(device)
        # logits_flat, hidden = rnn_forwardpass_spectrogram(model, spec, hidden)

        with torch.no_grad():
            logits_flat = model(specs, device)


        # Do something with the color palette

        ## Let's plot the most recent mel and the most recently generate palette
        mel = spec_t.reshape((128, 157, 1))
        print(logits_flat[:, -1, :] * 255)
        plot_palette_strip(logits_flat[:, -1, :], model.n_colors, mel)
        plt.show()


    p.terminate()
 

# when to resume recording again? immediately after inference or after some time has passed?
# how often to reset the hidden state?




    
