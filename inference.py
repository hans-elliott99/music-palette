
import random
import os
from pathlib import Path
import dill
import matplotlib.pyplot as plt

import torch
import utils
import dataload
import models.models_rnn as models_rnn


def load_prep_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model


@torch.no_grad()
def forwardpass_sequence(model, X, y, device):

    criterion = torch.nn.MSELoss()
    loss = 0
    rgb_dist = 0
    rgb_acc = 0
    preds = []

    seq_iters = X.shape[1]
    h0 = model.init_hidden(batch_size=1).to(device)
    for s in range(seq_iters):
            x_s = X[:, s, :]
            logits_flat, h0 = model(x_s, h0)
            logits_mat = logits_flat.reshape(logits_flat.shape[0], model.n_colors, 3) #(b, n_colors, rgb[3])
            preds.append(logits_mat)

            truth_mat = y[:, s, :model.n_colors]
            truth_flat  = truth_mat.view(truth_mat.size(0), -1) #(b, n_colors*3)

            # metrics
            loss += criterion(logits_flat, truth_flat)
            rgb_dist += utils.redmean_rgb_dist(logits_mat, truth_mat, scale_rgb=True)
            rgb_acc += utils.rgb_accuracy(logits_mat, truth_mat, scale_rgb=True, window_size=10)
    
    rgb_dist /= seq_iters
    rgb_acc  /= seq_iters
    preds = torch.stack(preds, dim=1) #return shape: batch[1], seq_length, n_colors, rgb[3]

    return {"mse":loss.item(), "rgb_dist":rgb_dist.item(), "rgb_acc":rgb_acc}, preds


def plot_pred_true_palettes(truth_arr, pred_arr):
    assert len(truth_arr.shape) == 2
    assert len(pred_arr.shape) == 2
    n_colors = pred_arr.shape[0]

    fig, axes = plt.subplots(n_colors, 2, figsize=(8,5))
    plt.subplots_adjust(hspace=0.5)

    fig.suptitle("True Color | Predicted Color")
    for i in range(n_colors):
        true_col = [int(c*255) for c in truth_arr[i].tolist()]
        pred_col = [int(c*255) for c in pred_arr[i].tolist()]

        ax1, ax2 = axes[i, 0], axes[i, 1]
        # ax1 = plt.subplot(n_colors, 2, i+1) ##subplot inds start at 1
        ax1.set_title("True RGB: " + ', '.join([str(c) for c in true_col]))
        ax1.imshow(utils.quick_color_display(true_col))

        # ax2 = plt.subplot(n_colors, 2, i+2)
        ax2.set_title("Pred RGB: " + ', '.join([str(c) for c in pred_col]))
        ax2.imshow(utils.quick_color_display(pred_col))

    
    plt.show()




if __name__=="__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    model_path = "./checkpoints/chkpnt_0.pth"
    model:models_rnn.SeqRNNConv = torch.load(model_path)
    model = model.to(device)
    model.eval()
    # hidden = model.init_hidden(batch_size=1)


    train_paths, valid_paths, test_paths = dataload.split_samples_by_song(
        arraydata_path="data_arrays_seq5", 
        audio_path    ="audio_wav",
        splitmeta_path="meta.json",
        valid_share   =0.1,
        random_seed   =123
        )
    test_dataset = dataload.SeqAudioRgbDataset(paths_list=test_paths,
                                               max_seq_length=5,
                                               data_dir="data_arrays_seq5"
                                               )


    X, y = test_dataset[ random.randint(0, len(test_dataset)-1) ]
    X = X.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device) #unsqueeze to add batch dimension

    metrics, preds = forwardpass_sequence(model, X, y, device)
    
    print(metrics)

    for i in range(preds.shape[1]): #sequence 
        plot_pred_true_palettes(y[0, i, :], preds[0, i, :])