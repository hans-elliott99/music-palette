import numpy as np
import torch
import os
from pathlib import Path
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.metrics import f1_score

#TODO[s] undone

# MISC ------------------------------------------------------------------------
def quick_color_display(rgb:list, h=30, w=30):
    """plt.imshow(quick_color_display([10, 123, 30]))
    """
    rgb = [round(v) for v in rgb]
    block = np.zeros((h, w, 3))
    block[:,:,:] = rgb
    return block.astype(np.uint8)


def wav_to_melspec(wav_path,
                    sample_rate=16000, n_mels=128, 
                    n_fft=2048, hop_length=512):
    # create mel-spectorgram
    y, sr = librosa.load(wav_path, sr=sample_rate)
    spec = librosa.feature.melspectrogram(y=y, sr=sr,
                                        n_mels=n_mels, 
                                        n_fft=n_fft,
                                        hop_length=hop_length)
    # convert to log scale (dB scale)
    log_spec = librosa.amplitude_to_db(spec, ref=1.)
    log_spec = log_spec[np.newaxis, ...] #add Channel dim
    return log_spec

def plot_predicted_melspec(true_melspec, pred_array, savename=None, **kwargs):
    """
    true_melspec: shape=(Channels[1], Height[mel-spec features], Width[time-steps])
    pred_array  : shape=(Batch Dim[optional], Number of Patches, Predicted Features)
    savename    : Str or None. If provided, save fig to image.
    kwargs      : include options for plt.figure
    """
    if len(pred_array.shape) == 3:
        n_patches = pred_array.shape[1]
    else:
        n_patches = pred_array.shape[0]
    mel_spec_feats = true_melspec.shape[1]
    timeunits_per_patch = int(math.prod(pred_array.shape) / (n_patches*mel_spec_feats))
    out_img = pred_array.reshape(n_patches, mel_spec_feats, timeunits_per_patch).detach()
    out_img = torch.concat([out_img[i] for i in range(out_img.shape[0])], axis=1)
    
    plt.figure(**kwargs)
    plt.subplot(211)
    plt.imshow(true_melspec.squeeze(0))
    plt.title("True Mel-Spec")
    plt.subplot(212)
    plt.imshow(out_img)
    plt.title("Generated Mel-Spec")
    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()


def get_audio_color(audio_file:str, audio_color_map_path:str, view=True):
    audio_file = audio_file.lower().replace("\\", "/")
    filename = audio_file.split("/")[-1]

    ac = pd.read_csv(audio_color_map_path)
    colors = []
    for i in range(ac.shape[1]-1):
        rgb = ac.loc[ac.audio_clip==filename, f"rgb_clust_{i}"].values[0]
        rgb = [int(c) for c in rgb.split()]
        colors.append(rgb)

    if view:
        color_seq = np.array([colors]) #(1, 5, 3)
        high_lum = [int(c) for c in pick_highest_luminance(color_seq)[0][0]]
        palette = np.vstack([quick_color_display(col) for col in colors])
        plt.imshow(palette)
        plt.axis("off")
        plt.title(filename)
        label_step = palette.shape[0] // len(colors)
        for i in range(len(colors)):
            plt.text(x = palette.shape[1]+3, y = (label_step)*(i+1),
             s = str(colors[i])+" (highest luminance)" if colors[i]==high_lum else str(colors[i])
             )
        plt.show()
    
    return colors


# METRICS ----------------------------------------------------------------------
class Metric:
    def __init__(self, fn, init_value=0., **fn_kwargs) -> None:
        self.calculate = fn
        self.fn_kwargs = fn_kwargs
        self.batch = RunningMean()
        self.accum_value = init_value

    def step(self, true, pred) -> None:
        self.accum_value += self.calculate(true, pred, **self.fn_kwargs)
    
    def batch_update(self, scale=1.) -> None:
        self.batch.update(self.accum_value * scale)
        self.accum_value = 0.

    def reset(self):
        self.accum_value = 0.
        self.batch.reset()


@torch.no_grad()
def class_accuracy(true_idx, preds, is_logits=False):
    if is_logits:
        preds = preds.max(axis=1).indices
    return (preds == true_idx).sum().item() / true_idx.shape[0]

def multiclass_f1score(true_idx, preds, average="macro", is_logits=False):
    preds = preds.detach().cpu()
    true_idx = true_idx.cpu()
    if is_logits:
        preds = preds.max(axis=1).indices
    return f1_score(true_idx, preds, average=average, zero_division=0)

@torch.no_grad()
def rgb_accuracy(truth_tensor, pred_tensor, window_size, scale_rgb=True):
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

@torch.no_grad()
def redmean_rgb_dist(truth_tensor:torch.tensor, pred_tensor:torch.tensor, scale_rgb=True):
    pred = pred_tensor.clone()
    true = truth_tensor.clone()
    #shapes: (batch_size, n_colors, rgb[3])
    if scale_rgb:
        pred *= 255
        true *= 255
    rmean = (pred[:, :, 0] + true[:, :, 0])/2
    r2 = torch.pow(pred[:, :, 0] - true[:, :, 0], 2)
    g2 = torch.pow(pred[:, :, 1] - true[:, :, 1], 2)
    b2 = torch.pow(pred[:, :, 2] - true[:, :, 2], 2)
    bit_shift = 2**8
    per_palette_loss = \
        torch.sqrt( ((512+rmean)*r2)/bit_shift + 4*g2 + ((767-rmean)*b2)/bit_shift )
    return torch.mean( torch.sum(per_palette_loss, axis=1) ).item()


# TRAINING UTILS --------------------------------------------------------------

#TODO: allow for patching in the time AND frequency direction - see https://arxiv.org/pdf/2110.05069v3.pdf
def create_patched_input(X:torch.tensor, n_patches:int=8, pad_method:str="zero", flatten:bool=True):
    """X.shape = (?, Channels, Height/Features, Width/Time) 
    Returns tensor of shape (?, n_patches, Channels, Height, feats_per_patch) where
    feats_per_patch = (W // n_patches) + 1 if W % n_patches != 0 , or
                    = W // n_patches if W % n_patches == 0  
    """
    device = X.device
    batched = True
    if len(X.shape) == 3:
        X = X.unsqueeze(0)
        batched = False
    b, C, H, W = X.shape
    feats_per_patch = W // n_patches
    pad_array = None
    if W % n_patches != 0:
        feats_per_patch += 1
        diff = feats_per_patch - (W - (feats_per_patch*(n_patches-1)))
        assert diff < feats_per_patch, f"The width {W} cannot be split into {n_patches} patches of constant length."

        if pad_method.lower().startswith("z"): #zero
            pad_array = torch.zeros((b, C, H, diff))
        elif pad_method.lower().startswith("me"): #mean
            pad_array = torch.full(size=(b, C, H, diff), fill_value=X.mean())
        elif pad_method.lower().startswith("mi"): #min
            pad_array = torch.full(size=(b, C, H, diff), fill_value=X.min())
        else:
            raise NotImplementedError("Argument pad_method must be one of 'zero', 'mean', or 'min'.")

    t = torch.empty((b, n_patches, C, H, feats_per_patch), device=device)
    for i in range(n_patches):
        patch = X[:,:,:, (i * feats_per_patch) : ((i+1)*feats_per_patch) ]
        if (i+1) == n_patches and pad_array is not None:
            pad_array = pad_array.to(device)
            patch = torch.cat([patch, pad_array], axis=-1)
        t[:, i] = patch
    
    if flatten:
        t = t.flatten(-3,-1) #(batch, n_patches, feats)
    if not batched:
        t = t.squeeze(0)
    return t


def pick_highest_luminance(y_array):
    """Calculates which RGB color has the highest perceived luminance.
    Selects (for each palette in a provided sequence) which color has the highest perceived luminance
    using the simple relative luminance formula as seen here:
    https://en.wikipedia.org/wiki/Relative_luminance
    Returns an array of shape (sequence_length, 1, 3) given (sequence_length, N_colors, 3).
    """
    no_seq = False
    if len(y_array.shape) == 2:
        no_seq = True
        y_array = y_array.unsqueeze(0) #add sequence/batch dimension if there is none
    y_out = torch.zeros((y_array.shape[0], 1, 3), dtype=torch.float32) # seq_len, 1 color, rgb
    lum_coef = torch.tensor([[0.2126], [0.7152], [0.0722]]) #3,1
    # mat mul w lum coefs to get the highest perceived luminance of all colors in palette (applied per-palette)
    y_lum = y_array @ lum_coef #5,5,3 @ 3,1 = 5,5,1
    inds = torch.argmax(y_lum, axis=1).flatten()
    for i, ix in enumerate(inds):
        y_out[i, :] = y_array[i, ix, :]
    if no_seq:
        y_out = y_out.squeeze(0)
    return y_out


def update_dropout_p(model:object, config:dict, verbose=True):
    dropouts = {"embed_dropout":None, "attn_dropout":None, "resid_dropout":None, "mlp_dropout":None, "head_dropout":None}
    for attr in dropouts.keys():
        old_val = getattr(model, attr)
        if old_val != config[attr]:
            setattr(model, attr, config[attr])
            if verbose:
                print(f"Updating {attr} from {old_val} to {config[attr]}.")
        dropouts[attr] = config[attr]

    for name, param in model.named_modules():
        if "embed_drop" in name or "proj_drop" in name:
            param.p = dropouts["embed_dropout"]
        elif "attn_drop" in name:
            param.p = dropouts["attn_dropout"]
        elif "resid_drop" in name:
            param.p = dropouts["resid_dropout"]
        elif "mlp_drop" in name or "mlp.net.3" in name:   #TODO: mlp.net.3 is left for back-compat.
            param.p = dropouts["mlp_dropout"]
        elif "head_drop" in name:
            param.p = dropouts["head_dropout"]
    return model


def save_model_with_shape(model, 
                          save_dir, save_name="checkpoint", 
                          config_file=None, config=None):

    os.makedirs(save_dir, exist_ok=True)
    save_name = save_name.split(".")[0] + ".pth"
    
    if config_file is not None:
        config_file = Path(save_dir) / Path(config_file.split(".")[0]+".json")
        with open(config_file, "w") as f:
            json.dump(config, f)
    
    torch.save(model, Path(save_dir) / Path(save_name))


class RunningMean:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.last  = 0
        self.count = 0
        self.sum   = 0
        self.mean  = 0

    def update(self, value, n=1):
        self.last   = value
        self.count += n
        self.sum   += value*n
        self.mean   = self.sum / self.count


class SimpleLog:
    def __init__(self, filepath, reset=True) -> None:
        self.file = filepath
        if reset:
            self.reset()
    
    def append(self, step, value):
        """Append a row to the log file"""
        with open(self.file, 'a+') as f:
            f.write(f"{step}; {value}\n")

    def reset(self):
        """Empty the log file if it already exists."""
        open(self.file, 'w').close()

class JsonLog:
    def __init__(self, filepath, reset=True) -> None:
        self.file = filepath
        if reset:
            self.reset()

    def reset(self):
        """Empty the log file if it already exists."""
        open(self.file, 'w').close()

    def write(self, **kwargs):
        """Write any number of key,value pairs to the log file."""
        with open(self.file, "w") as fp:
            json.dump(kwargs, fp)


def load_json(file):
    with open(file, "r") as fp:
        o = json.load(fp)
    return o




def lr_linearwarmup_cosinedecay(iter_1_based, max_lr, min_lr, warmup_iters, decay_iters):
    """https://github.com/karpathy/nanoGPT/blob/master/train.py
    """
    # do a linear warmup to max_lr for the first given iters
    if iter_1_based < warmup_iters:
        return max_lr * iter_1_based / warmup_iters
    
    # return the min. lr if training continues past specified decay iters
    if iter_1_based > decay_iters:
        return min_lr

    # otherwise, do a cosine decay down to min-lr    
    decay_ratio = (iter_1_based - warmup_iters) / (decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1, "lr_linearwarmup_cosinedecay: bad decay_ratio"
    coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #range 0, 1
    
    return min_lr + coef * (max_lr - min_lr)



# if __name__=="__main__":

#     for i in range(270, 275):
#         get_audio_color(f"temp/{i}_audioclip_0.wav", "audio_color_mapping.csv")