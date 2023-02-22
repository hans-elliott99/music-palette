
import random
import os
from pathlib import Path
import dill
import numpy as np
import matplotlib.pyplot as plt

import torch
import src.utils as utils
import src.dataload as dataload
from src.models.transformers import ConvTransformer, PatchTransformer, GPTPatchTransformer
from src.dataload import SeqAudioRgbDataset
from tqdm import tqdm

@torch.no_grad()
def batched_testloop(test_dataloader, model, criterion, device):
    
    model.eval()
    losses, accs, dists = (utils.RunningMean() for _ in range(3))
    
    for i, (X, y) in tqdm(enumerate(test_dataloader)):

        X = X.to(device)
        y = y.to(device)

        batch_size = X.shape[0]
        seq_len    = X.shape[1]

        # FORWARD PASS
        logits_flat = model(X) #(b, T, n_colors*3)
        logits_mat = logits_flat.reshape(batch_size, seq_len, model.n_colors, 3) #(b, T, n_colors, rgb[3])

        truth_mat = y[:,:, :model.n_colors, :]   #(b, T, n_colors, 3)
        truth_flat = truth_mat.reshape(batch_size, seq_len, -1)  #(b, T, n_colors*3)
        loss = criterion(logits_flat, truth_flat)

        # Calc metrics
        rgb_dist, rgb_acc = 0, 0
        for i in range(seq_len):
            rgb_dist += utils.redmean_rgb_dist(logits_mat[:, i, :], truth_mat[:, i, :], scale_rgb=True)
            rgb_acc  += utils.rgb_accuracy(logits_mat[:, i, :], truth_mat[:, i, :], scale_rgb=True, window_size=10)
        
        # Track metrics
        losses.update(loss.item())
        accs.update(rgb_acc/seq_len)
        dists.update(rgb_dist.item()/seq_len)
    
    return losses, dists, accs

@torch.no_grad()
def forward_seq_example(X, y, model):
    model.eval()
    batch_size = X.shape[0] #==1
    seq_len    = X.shape[1]

    # FORWARD PASS
    logits_flat = model(X) #(b, T, n_colors*3)
    logits_mat = logits_flat.reshape(batch_size, seq_len, model.n_colors, 3) #(b, T, n_colors, rgb[3])

    truth_mat = y[:,:, :model.n_colors, :]   #(b, T, n_colors, 3)
    truth_flat = truth_mat.reshape(batch_size, seq_len, -1)  #(b, T, n_colors*3)
    loss = torch.nn.functional.mse_loss(logits_flat, truth_flat)

    # Calc metrics
    rgb_dist, rgb_acc = 0, 0
    for i in range(seq_len):
        rgb_dist += utils.redmean_rgb_dist(logits_mat[:, i, :], truth_mat[:, i, :], scale_rgb=True)
        rgb_acc  += utils.rgb_accuracy(logits_mat[:, i, :], truth_mat[:, i, :], scale_rgb=True, window_size=10)

    return logits_mat, loss.item(), rgb_dist.item(), rgb_acc

@torch.no_grad()
def forward_patched_example(X, y, model):
    model.eval()

    # just use first tensors in the sequence
    X = utils.create_patched_input(X[:,0], n_patches=model.n_patches, pad_method="min")
    y = y[:,0]

    batch_size = X.shape[0] #==1
    n_patches  = X.shape[1] 

    # FORWARD PASS
    logits_flat = model(X) #(b, n_colors*3)
    logits_mat = logits_flat.reshape(batch_size, model.n_colors, 3) #(b, n_colors, rgb[3])

    truth_mat = y[:, :model.n_colors, :]   #(b, n_colors, 3)
    truth_flat = truth_mat.reshape(batch_size, -1)  #(b, n_colors*3)
    loss = torch.nn.functional.mse_loss(logits_flat, truth_flat)

    # Calc metrics
    rgb_dist = utils.redmean_rgb_dist(logits_mat, truth_mat, scale_rgb=True)
    rgb_acc  = utils.rgb_accuracy(logits_mat, truth_mat, scale_rgb=True, window_size=10)

    return logits_mat, loss.item(), rgb_dist.item(), rgb_acc

# ----------------------------------------------------------------------------------------
def plot_pred_true_palettes(truth_arr, pred_arr):
    assert len(truth_arr.shape) == 2
    assert len(pred_arr.shape) == 2
    n_colors = pred_arr.shape[0]

    fig, axes = plt.subplots(n_colors+1, 2, figsize=(8,5))
    plt.subplots_adjust(hspace=0.5)

    fig.suptitle("True Color | Predicted Color")
    for i in range(n_colors):
        true_col = [int(c*255) for c in truth_arr[i].tolist()]
        pred_col = [int(c*255) for c in pred_arr[i].tolist()]

        ax1, ax2 = axes[i, 0], axes[i, 1]
        ax1.set_title("True RGB: " + ', '.join([str(c) for c in true_col]))
        ax1.imshow(utils.quick_color_display(true_col))

        ax2.set_title("Pred RGB: " + ', '.join([str(c) for c in pred_col]))
        ax2.imshow(utils.quick_color_display(pred_col))
    plt.show()

def plot_melspec_seq(X):
    if len(X.shape) == 5:
        X = X.squeeze(0) #rm batch dim
    
    o = np.concatenate(np.array(X.cpu()), axis=1).squeeze(0) #concatenate on the height dimension
    o = o[..., np.newaxis] #channel dim must be last
    plt.imshow(o)
    plt.show()

if __name__=="__main__":
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("device:",device)

    model_path = "./checkpoints/gpt_base.pth"
    model = torch.load(model_path)
    model = model.to(device)


    train_paths, valid_paths, test_paths = dataload.split_samples_by_song(
        arraydata_dir="data_arrays_seq5", 
        audio_dir    ="./data/pop_videos/audio_wav",
        splitmeta_path="./data/pop_videos/train_test_songs.json",
        valid_share   =0.1,
        random_seed   =123
        )
    test_dataset = SeqAudioRgbDataset(paths_list=test_paths,
                                     max_seq_length=5,
                                     data_dir="./data_arrays_seq5",
                                     y_transform=utils.pick_highest_luminance
                                     )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 10,
        shuffle = False,
        num_workers = 1,
        drop_last = False,
        pin_memory = True if torch.cuda.is_available() else False
    )


    if False: #test the entire hold-out data
        loss, dist, acc = batched_testloop(
            test_dataloader, model,
            criterion=torch.nn.MSELoss(),
            device=device
        )
        print(
            "Test Performance:"
            f"\n\tMean Loss = {loss.mean :.5f}"
            f"\n\tMean RGB Dist = {dist.mean :.5f}"
            f"\n\tMean RGB Acc = {acc.mean :.5f}"
        )
    
    # test a random example
    X, y = test_dataset[ random.randint(0, len(test_dataset)-1) ]
    X = X.unsqueeze(0).to(device) #[0] since using sequential dataset
    y = y.unsqueeze(0).to(device) #unsqueeze to add batch dimension

    if isinstance(model, (PatchTransformer, GPTPatchTransformer)):
        preds, loss, rgb_dist, rgb_acc = forward_patched_example(X, y, model)


    print(preds*255)
    print(y[0,0]*255)
    print(f"Test example loss = {loss}")
    print(f"Test example RGB Dist = {rgb_dist}")
    print(f"Test example RGB Accuracy = {rgb_acc}")

    plot_melspec_seq(X)
    print(y.shape, preds.shape)
    plot_pred_true_palettes(y[0, 0], preds[0])
