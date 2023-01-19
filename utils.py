import numpy as np
import torch
import os
from pathlib import Path
import json

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

@torch.no_grad()
def redmean_rgb_dist(pred_tensor:torch.tensor, truth_tensor:torch.tensor, scale_rgb=True):
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
    return torch.mean( torch.sum(per_palette_loss, axis=1) )


def quick_color_display(rgb:list, h=30, w=30):
    """plt.imshow(quick_color_display([10, 123, 30]))
    """
    rgb = [round(v) for v in rgb]
    block = np.zeros((h, w, 3))
    block[:,:,:] = rgb
    return block.astype(np.uint8)

 


#import colorsys
#colorsys.rgb_to_hsv


def save_model_with_shape(model, save_dir, save_name="checkpoint", config_file=None, config=None):

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




