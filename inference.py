
import random
import os
from pathlib import Path
import dill
import matplotlib.pyplot as plt

import torch
import src.utils as utils
import src.dataload as dataload
from src.models.transformers import ConvTransformer
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
def forward_example():
    pass


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
    print("device:",device)

    model_path = "./checkpoints/trfmr_highlum.pth"
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


    loss, dist, acc = batched_testloop(
        test_dataloader, model,
        criterion=torch.nn.MSELoss(),
        device=device
    )


    print(
        "Test Performance:"
        f"\nMean Loss = {loss.mean :.5f}"
        f"\nMean RGB Dist = {dist.mean :.5f}"
        f"\nMean RGB Acc = {acc.mean :.5f}"
    )
    # X, y = test_dataset[ random.randint(0, len(test_dataset)-1) ]
    # X = X.unsqueeze(0).to(device)
    # y = y.unsqueeze(0).to(device) #unsqueeze to add batch dimension
    # metrics, preds = forwardpass_sequence(model, X, y, device)
    


    # for i in range(preds.shape[1]): #sequence 
    #     plot_pred_true_palettes(y[0, i, :], preds[0, i, :])




# @torch.no_grad()
# def rnn_forwardpass_sequence(model, X, y, device):

#     criterion = torch.nn.MSELoss()
#     loss = 0
#     rgb_dist = 0
#     rgb_acc = 0
#     preds = []

#     seq_iters = X.shape[1]
#     h0 = model.init_hidden(batch_size=1).to(device)
#     for s in range(seq_iters):
#             x_s = X[:, s, :]
#             logits_flat, h0 = model(x_s, h0)
#             logits_mat = logits_flat.reshape(logits_flat.shape[0], model.n_colors, 3) #(b, n_colors, rgb[3])
#             preds.append(logits_mat)

#             truth_mat = y[:, s, :model.n_colors]
#             truth_flat  = truth_mat.view(truth_mat.size(0), -1) #(b, n_colors*3)

#             # metrics
#             loss += criterion(logits_flat, truth_flat)
#             rgb_dist += utils.redmean_rgb_dist(logits_mat, truth_mat, scale_rgb=True)
#             rgb_acc += utils.rgb_accuracy(logits_mat, truth_mat, scale_rgb=True, window_size=10)
    
#     rgb_dist /= seq_iters
#     rgb_acc  /= seq_iters
#     preds = torch.stack(preds, dim=1) #return shape: batch[1], seq_length, n_colors, rgb[3]

#     return {"mse":loss.item(), "rgb_dist":rgb_dist.item(), "rgb_acc":rgb_acc}, preds
