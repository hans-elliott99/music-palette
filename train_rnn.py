
import time
import random
import wandb 
from dotenv import dotenv_values

import numpy as np
import torch
import torch.nn as nn
import models.models_rnn as models_rnn
import utils
import dataload



def batched_trainloop(train_dataloader, model, criterion, optimizer, config, device, batch_print_freq=10):
    
    model.train()

    losses, accs, dists, batch_times = (utils.RunningMean() for _ in range(4))

    for i, (X, y) in enumerate(train_dataloader):
        batch_loss = 0
        rgb_dist, rgb_acc = 0, 0
        time0 = time.time()

        # Put data on device
        X = X.to(device)
        y = y.to(device)

        # FORWARD PASS
        seq_iters = X.shape[1] #max_seq_length 
        
        # inner-loop: process each (batch) of the sequences step-by-step
        h0 = model.init_hidden(batch_size=X.shape[0]).to(device)
        for step in range(seq_iters):
            x_s = X[:, step, :]
            logits_flat, h0 = model(x_s, h0)
            logits_mat = logits_flat.reshape(logits_flat.shape[0], model.n_colors, 3) #(b, n_colors, rgb[3])
           
            truth_mat = y[:, step, :model.n_colors]
            truth_flat  = truth_mat.view(truth_mat.size(0), -1) #(b, n_colors*3)
           
            batch_loss += criterion(logits_flat, truth_flat)

            # metrics
            rgb_dist += utils.redmean_rgb_dist(logits_mat, truth_mat, scale_rgb=True)
            rgb_acc += utils.rgb_accuracy(logits_mat, truth_mat, scale_rgb=True, window_size=10)

        # BACKPROP (through time)
        batch_loss.backward(retain_graph=True)
        h0.detach_() ##detach hidden state

        # OPTIMIZE
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Track metrics
        losses.update(batch_loss.item())
        accs.update(rgb_acc/seq_iters)
        dists.update(rgb_dist.item()/seq_iters)
        batch_times.update(time.time() - time0)

        if i % batch_print_freq == 0 or i==len(train_dataloader)-1:
            print(f" [Batch {i+1}/{len(train_dataloader)} (et={batch_times.last :.2f}s)]")
            print(f" \tLoss={losses.last :.5f} \tRGBdist={dists.last :.5f} \tAcc={accs.last :.5f}")

    return losses, dists, accs 


@torch.no_grad()
def batched_validloop(valid_dataloader, model, criterion, config, device):
    
    model.eval()

    losses, accs, dists = (utils.RunningMean() for _ in range(3))
    for i, (X, y) in enumerate(valid_dataloader):
        rgb_dist = 0
        rgb_acc = 0
        batch_loss = 0

        X = X.to(device)
        y = y.to(device)

        # FORWARD
        seq_iters = X.shape[1] #max_seq_length 
        
        # inner-loop: process each (batch) of the sequences step-by-step
        h0 = model.init_hidden(batch_size=X.shape[0]).to(device)
        for s in range(seq_iters):
            x_s = X[:, s, :]
            logits_flat, h0 = model(x_s, h0)
            logits_mat = logits_flat.reshape(logits_flat.shape[0], model.n_colors, 3) #(b, n_colors, rgb[3])

            truth_mat = y[:, s, :model.n_colors]
            truth_flat  = truth_mat.view(truth_mat.size(0), -1) #(b, n_colors*3)
            
            batch_loss += criterion(logits_flat, truth_flat)

            # metrics
            rgb_dist += utils.redmean_rgb_dist(logits_mat, truth_mat, scale_rgb=True)
            rgb_acc += utils.rgb_accuracy(logits_mat, truth_mat, scale_rgb=True, window_size=10)
            # preds.append(logits_mat)

        # Track metrics
        losses.update(batch_loss.item())
        accs.update(rgb_acc/seq_iters)
        dists.update(rgb_dist.item()/seq_iters)
    
    return losses, dists, accs


if __name__=="__main__":
    # User Control
    seed = 87271
    DATA_DIR       = "./data_arrays_seq5"
    AUDIO_DIR      = "audio_wav"
    SPLITMETA_PATH = "./train_test_songs.json"
    
    LOG_WANDB              = True

    LOAD_CHECKPOINT        = "./checkpoints/chkpnt_1col.pth"
    LOAD_CHECKPOINT_CONFIG = "./checkpoints/chkpnt_1col_config.json"
    SAVE_CHECKPOINT        = True
    MODEL_SAVE_NAME        = "chkpnt_1col"

    config = {
        # Model specs
        "n_colors"        : 1,
        "max_seq_length"  : 5,
        "dropout"         : 0.5,
        "gru_hidden_size" : 64,

        # Training specs
        "batch_size" : 8,
        "epochs"     : 100,
        "last_epoch" : 0,  #will be overriden by checkpoint if one is loaded

        # Optimizer
        "lr" : 3e-4,
    }

    # -------------------------SETUP------------------------- #
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    torch.cuda.empty_cache()

    wandb_key = dotenv_values(".env")["WANDB_KEY"]
    if LOG_WANDB:
        wandb.login(key=wandb_key)
        wandb.init(
            project="MusicPalette",
            config=config,
            group="Seq_RNN_CNN",
            anonymous="allow"
        )

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print("CUDA:", use_cuda)


    # -------------------------PREPARE DATA------------------------- #
    train_paths, valid_paths, test_paths = dataload.split_samples_by_song(
        arraydata_path=DATA_DIR, 
        audio_path    =AUDIO_DIR,
        splitmeta_path=SPLITMETA_PATH,
        valid_share   =0.1,
        random_seed   =seed
        )
    print("Train Samples:", len(train_paths),
          "\nValid Samples:", len(valid_paths))
    
    train_dataset = dataload.SeqAudioRgbDataset(paths_list=train_paths,
                                                max_seq_length=config["max_seq_length"],
                                                data_dir=DATA_DIR
                                                )
                                        
    valid_dataset = dataload.SeqAudioRgbDataset(paths_list=valid_paths,
                                                max_seq_length=config["max_seq_length"],
                                                data_dir=DATA_DIR
                                                )


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config["batch_size"],
        shuffle = True,
        num_workers = 1,
        drop_last = True,
        pin_memory = True if use_cuda else False
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = config["batch_size"],
        shuffle = True,
        num_workers = 1,
        drop_last = False,
        pin_memory = True if use_cuda else False
    )

    # -------------------------PREPARE MODEL------------------------- #
    start_epoch = 0
    if not LOAD_CHECKPOINT:
        ex_x, _ = train_dataset[0]
        model = models_rnn.SeqRNNConv(
            X_shape=ex_x.shape,
            n_colors=config["n_colors"],
            gru_hidden_size=config["gru_hidden_size"],
            dropout=config["dropout"]
        )
    else:
        print(f"Loading model checkpoint: {LOAD_CHECKPOINT}")
        model = torch.load(LOAD_CHECKPOINT)
        config_old = utils.load_json(LOAD_CHECKPOINT_CONFIG)
        start_epoch = config_old['last_epoch'] + 1


    model = model.to(device)
    print(f"Model params: {model.n_params :,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    if LOG_WANDB:
        wandb.watch(model)

    # -------------------------TRAIN LOOP------------------------- #
    ep_losses, ep_dists, ep_accs = [], [], []
    val_losses, val_dists, val_accs = [], [], []

    try:
        for ep in range(start_epoch, start_epoch+config["epochs"]):

            time0 = time.time()
            print(f"\nEPOCH {ep+1} / {start_epoch+config['epochs']}")
            
            el, ed, ea = batched_trainloop(train_dataloader, 
                                           model=model, 
                                           criterion=criterion, 
                                           optimizer=optimizer,
                                           config=config, 
                                           device=device,
                                           batch_print_freq=10)
            ep_losses.append(el.mean)
            ep_dists.append(ed.mean)
            ep_accs.append(ea.mean)

            print(f" * Epoch Means: ",
                f"Loss={el.mean :.5f} \tRGBDist={ed.mean :.5f} \tRGBAcc={ea.mean :.5f}")

            vl, vd, va = batched_validloop(valid_dataloader,
                                           model=model, 
                                           criterion=criterion,
                                           config=config,
                                           device=device)
            val_losses.append(vl.mean)
            val_dists.append(vd.mean)
            val_accs.append(va.mean)

            print(f" * Valid Means: "
                f"Loss={vl.mean :.5f} \tRGBDist={vd.mean :.5f} \tRGBAcc={va.mean :.5f}")

            stp_time = time.time() - time0
            print(f" * Epoch+Valid Time: {stp_time :.1f}s (Est. Train Length = {(stp_time * config['epochs']) / (60*60):.1f} hrs)")

            if LOG_WANDB:
                try:
                    wandb.log({
                        "loss"         : el.mean,
                        "val_loss"     : vl.mean,
                        "rgb_dist"     : ed.mean,
                        "val_rgb_dist" : vd.mean,
                        "acc"          : ea.mean,
                        "val_acc"      : va.mean
                    })
                except Exception as e:
                    print("[wandb.log error]", e)
        config["last_epoch"] = ep

    except KeyboardInterrupt as e:
        i = input("Training interrupred. Save model checkpoint? (y/n): ")
        if i.lower().startswith("y"):
            SAVE_CHECKPOINT = True
            config["last_epoch"] = ep
            print("Saving.")
        elif i.lower().startswith("n"):
            SAVE_CHECKPOINT = False



    # Save metrics for analysis
    log = utils.JsonLog("./checkpoints/metricslog.json")
    log.write(train_loss=ep_losses, train_rgbdist=ep_dists, train_acc=ep_accs,
              val_loss=val_losses, val_rgbdists=val_dists, val_acc=val_accs)
    
    # Save model and configuration
    if SAVE_CHECKPOINT:
        utils.save_model_with_shape(model, 
                                    save_dir="./checkpoints",
                                    save_name=MODEL_SAVE_NAME, 
                                    config_file=f"{MODEL_SAVE_NAME}_config.json", 
                                    config=config)