
# For recording of original data (and conversion to mel-spec)

import numpy as np
import dill
import pyaudio
import wave
import librosa

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
device_index = 0
save_path = "./recorded_data/silence_spec"


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



def convert_to_melspec(wav_filename,
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
    return log_spec[np.newaxis, ...] #add Channel dim



if __name__=="__main__":

    # Record audio
    p = pyaudio.PyAudio()

    stream = p.open(rate=RATE, 
                    channels=CHANNELS,
                    format=FORMAT,
                    input=True,
                    input_device_index=device_index)

    print("recording...")
    record_next_clip(stream, p.get_sample_size(FORMAT), "temp.wav")
    print("done.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert audio to mel-spec
    spec = convert_to_melspec("temp.wav")
    with open(save_path, 'wb') as f:
            dill.dump(spec, f)






    
