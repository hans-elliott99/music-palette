#!/usr/bin/env python
import os
import sys
import signal
import atexit
import time
import random
import wandb 
from dotenv import dotenv_values

import numpy as np
import torch
import torch.nn as nn

import src.utils as utils
from src.dataload import SeqAudioRgbDataset, split_samples_by_song
from src.models.transformers import ConvTransformer, PatchTransformer
import src.models.configs as configs


#TODO:
# PatchTransformer doesnt need data to be stored in sequential format, since each sample is a single melspec -> palette
# However, while the data is stored this way, could experiment with learnable memory across samples...


def convtransformer_update(X, y, model, criterion, optimizer=None):

    batch_size = X.shape[0]
    seq_len    = X.shape[1]

    # FORWARD PASS
    logits_flat = model(X) #(b, T, n_colors*3)
    logits_mat = logits_flat.reshape(batch_size, seq_len, model.n_colors, 3) #(b, T, n_colors, rgb[3])

    truth_mat = y[:,:, :model.n_colors, :]   #(b, T, n_colors, 3)
    truth_flat = truth_mat.reshape(batch_size, seq_len, -1) #(b, T, n_colors*3)
    loss = criterion(logits_flat, truth_flat)

    # BACKWARDS
    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Calc metrics
    rgb_dist, rgb_acc = 0, 0
    for i in range(seq_len):
        rgb_dist += utils.redmean_rgb_dist(logits_mat[:, i, :], truth_mat[:, i, :], scale_rgb=True)
        rgb_acc  += utils.rgb_accuracy(logits_mat[:, i, :], truth_mat[:, i, :], scale_rgb=True, window_size=10)

    return loss.item(), rgb_acc/seq_len, rgb_dist.item()/seq_len


def patchtransformer_update(X, y, model, criterion, config, optimizer=None):

    batch_size = X.shape[0]

    # FORWARD PASS
    x = utils.create_patched_input(X, 
                                   n_patches=config["n_patches"], 
                                   pad_method="min") #(b, n_patches, feats)

    logits_flat = model(x) #(b, n_colors*3)     
    logits_mat = logits_flat.reshape(batch_size, model.n_colors, 3) #(b, n_colors, rgb[3])

    truth_mat = y[:, :model.n_colors, :]   #(b, n_colors, 3)
    truth_flat = truth_mat.reshape(batch_size, -1) #(b, n_colors*3)
    loss = criterion(logits_flat, truth_flat)

    # BACKWARDS
    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Calc metrics
    rgb_dist = utils.redmean_rgb_dist(logits_mat, truth_mat, scale_rgb=True)
    rgb_acc  = utils.rgb_accuracy(logits_mat, truth_mat, scale_rgb=True, window_size=10)
    return loss.item(), rgb_acc, rgb_dist.item()


def batched_trainloop(train_dataloader, model, criterion, optimizer, config, device, batch_print_freq=10):
    
    model.train()
    losses, accs, dists, batch_times = (utils.RunningMean() for _ in range(4))

    for step, (X, y) in enumerate(train_dataloader):

        time0 = time.time()

        # Put data on device
        X = X.to(device)
        y = y.to(device)

        if isinstance(model, ConvTransformer):
            loss, rgb_acc, rgb_dist = convtransformer_update(X, y, model, criterion, optimizer)

        elif isinstance(model, PatchTransformer):
            loss, rgb_acc, rgb_dist = 0, 0, 0
            for i in range(X.shape[1]):
                # while using the sequentially stored data, process each melspec in the sequence as independent sample
                # (So really, one "batch" from dataloader is 5 batches)
                loss_, rgbacc_, rgbdist_ = patchtransformer_update(X[:,i,:], y[:,i,:], model, criterion, config, optimizer)
                loss     += loss_/X.shape[1]
                rgb_acc  += rgbacc_/X.shape[1]
                rgb_dist += rgbdist_/X.shape[1]

        # Track metrics
        losses.update(loss)
        accs.update(rgb_acc)
        dists.update(rgb_dist)
        batch_times.update(time.time() - time0)

        if (step+1) % batch_print_freq == 0 or step in [0, len(train_dataloader)-1]:
            print(f" [Batch {step+1}/{len(train_dataloader)} (et={batch_times.last :.2f}s)]")
            print(f" \tLoss={losses.last :.5f} \tRGBdist={dists.last :.5f} \tAcc={accs.last :.5f}")

    return losses, dists, accs


@torch.no_grad()
def batched_validloop(valid_dataloader, model, criterion, config, device):
    
    model.eval()
    losses, accs, dists = (utils.RunningMean() for _ in range(3))
    
    for i, (X, y) in enumerate(valid_dataloader):

        X = X.to(device)
        y = y.to(device)

        if isinstance(model, ConvTransformer):
            loss, rgb_acc, rgb_dist = convtransformer_update(X, y, model, criterion)

        elif isinstance(model, PatchTransformer):
            loss, rgb_acc, rgb_dist = 0, 0, 0
            for i in range(X.shape[1]):
                # while using the sequentially stored data, process each melspec in the sequence as independent sample
                loss_, rgbacc_, rgbdist_ = patchtransformer_update(X[:, i, :], y[:, i, :], model, criterion, config)
                loss     += loss_/X.shape[1]
                rgb_acc  += rgbacc_/X.shape[1]
                rgb_dist += rgbdist_/X.shape[1]

        # Track metrics
        losses.update(loss)
        accs.update(rgb_acc)
        dists.update(rgb_dist)
    
    return losses, dists, accs

accepted_models = {
    1 : "ConvTransformer",
    2 : "PatchTransformer"
}


if __name__=="__main__":
    # -------------------------USER INPUT------------------------- #
    seed = 87271
    DATA_DIR       = "./data_arrays_seq5"
    AUDIO_DIR      = "./data/pop_videos/audio_wav"
    SPLITMETA_PATH = "./data/pop_videos/train_test_songs.json"
    
    LOG_WANDB              = True
    SAVE_CHECKPOINT        = True

    MODEL_SAVE_NAME        = None or wandb.util.generate_id()
    LOAD_CHECKPOINT        = f"./checkpoints/{MODEL_SAVE_NAME}.pth" #will be created if doesnt exist
    LOAD_CHECKPOINT_CONFIG = f"./checkpoints/{MODEL_SAVE_NAME}_config.json"

    MODEL_CLASS  = PatchTransformer
    MODEL_CONFIG = configs.PatchTrfmrRegularizeConfig
    DATALOADER   = SeqAudioRgbDataset

    epochs = 500
    batch_size = 32
    config = {
        # Training specs
        "batch_size" : batch_size,
        "epochs"     : epochs,
        "last_epoch" : 0,  #will be overriden by checkpoint if one is loaded
        "max_seq_len" : 5, #data is currently stored as such    TODO max_seq_len is really a ConvTransformer arg and part of its config, only needed here due to SeqAudioRGBDataset

        # Optimizer / LR
        "lr" : 3e-4, #also the max lr if decay_lr==True
        "betas" : (0.9, 0.999),
        "weight_decay" : 3e-5,
        "decay_lr" : True,
        "min_lr" : 3e-5,  #should be ~ lr / 10
        "lr_warmup_steps" : 10,
        "lr_decay_steps"  : epochs - 10,
        "manual_new_lr"   : None, #set None to ignore
    }
    config = { **MODEL_CONFIG().to_dict(), **config }

    y_transform = None
    if config["n_colors"] == 1:
        y_transform = utils.pick_highest_luminance

    # X_transform = None
    # if config["n_patches"]:
    #     assert MODEL_CLASS == PatchTransformer
    #     X_transform = lambda x: utils.create_patched_input(x, config["n_patches"], pad_method="min")

    # -------------------------SETUP RUN------------------------- #
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    torch.cuda.empty_cache()

    wandb_key = dotenv_values(".env")["WANDB_KEY"]
    if LOG_WANDB:
        wandb.login(key=wandb_key)
        wandb.init(
            id=MODEL_SAVE_NAME,
            resume="allow",
            project="MusicPalette",
            config=config,
            group="PatchTransformer",
            anonymous="allow"
        )

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(time.strftime(r"%Y-%m-%d %H:%M"))
    print("CUDA:", use_cuda)

    # -------------------------PREPARE DATA------------------------- #
    train_paths, valid_paths, test_paths = split_samples_by_song(
        arraydata_dir =DATA_DIR, 
        audio_dir     =AUDIO_DIR,
        splitmeta_path=SPLITMETA_PATH,
        valid_share   =0.1,
        test_share    =0.2,
        random_seed   =seed
    )
    
    train_dataset = DATALOADER(paths_list=train_paths,                      #TODO dont need sequential dataset for patched model
                               data_dir=DATA_DIR,
                               max_seq_length=config.get("max_seq_len"),
                               y_transform=y_transform
                               )

    valid_dataset = DATALOADER(paths_list=valid_paths,
                               data_dir=DATA_DIR,
                               max_seq_length=config.get("max_seq_len"),
                               y_transform=y_transform
                               )
    ##avoid padding for now, just remove short seqs
    train_dataset.remove_short_seqs() 
    valid_dataset.remove_short_seqs()

    print("Train Samples:", len(train_dataset), 
        "\nValid Samples:", len(valid_dataset),
        "\nBatch Size:", config["batch_size"])

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
    model_exists = False
    if LOAD_CHECKPOINT is not None and os.path.exists(LOAD_CHECKPOINT):
        model_exists = True

    start_epoch = 0
    if not LOAD_CHECKPOINT or not model_exists:
        print(f"Preparing new model: {MODEL_SAVE_NAME}.")
        ex_x, _ = train_dataset[0]
        if MODEL_CLASS == PatchTransformer:                                 #TODO clean the x_shape init
            ex_x = utils.create_patched_input(ex_x, n_patches=config["n_patches"])
        model = MODEL_CLASS(X_shape=ex_x.shape, config=MODEL_CONFIG())
    else:
        print(f"Loading model checkpoint: {LOAD_CHECKPOINT}")
        model = torch.load(LOAD_CHECKPOINT)
        config_old = utils.load_json(LOAD_CHECKPOINT_CONFIG)
        start_epoch = config_old['last_epoch'] + 1


    model = model.to(device)
    print(f"Model params: {model.n_params :,}")


    # -------------------------LOSS & OPTIM------------------------- #
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config["lr"],
                                  betas=config['betas'],
                                  weight_decay=config['weight_decay']
                                  )

    # -------------------------TRAIN LOOP------------------------- #
    saved_successfully = False
    if LOG_WANDB:
        wandb.watch(model)

    ep_losses, ep_dists, ep_accs = [], [], []
    val_losses, val_dists, val_accs = [], [], []
    lr = config['lr']

    try:
        for ep in range(start_epoch, start_epoch+config["epochs"]):

            if config['decay_lr']:
                # Update lr: pass in the current iter (1-based), relative
                # to starting iter. 
                lr = utils.lr_linearwarmup_cosinedecay(
                    (ep+1)-start_epoch, 
                    max_lr=config['lr'],
                    min_lr=config['min_lr'],
                    warmup_iters=config['lr_warmup_steps'],
                    decay_iters=config['lr_decay_steps']
                    )
                optimizer.param_groups[0]['lr'] = lr
            elif config["manual_new_lr"] is not None:
                lr = config["manual_new_lr"]
                optimizer.param_groups[0]['lr'] = lr

            time0 = time.time()
            print(f"\nEPOCH {ep+1} / {start_epoch+config['epochs']} (lr={lr :.3E})")
            
            el, ed, ea = batched_trainloop(train_dataloader, 
                                           model=model, 
                                           criterion=criterion, 
                                           optimizer=optimizer,
                                           config=config, 
                                           device=device,
                                           batch_print_freq=15)
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
                wandb.log({
                    "loss"         : el.mean, "val_loss"     : vl.mean,
                    "rgb_dist"     : ed.mean, "val_rgb_dist" : vd.mean,
                    "acc"          : ea.mean, "val_acc"      : va.mean,
                    "lr"           : lr
                })

            if (ep+1) % 100 == 0:
                print("Saving model checkpoint.")
                config["last_epoch"] = ep
                utils.save_model_with_shape(model, 
                                save_dir="./checkpoints",
                                save_name=f"{MODEL_SAVE_NAME}_ep{ep}", 
                                config_file=f"{MODEL_SAVE_NAME}_config.json", 
                                config=config)
            #end of epoch iteration -----

        config["last_epoch"] = ep
    except KeyboardInterrupt as e:
        i = input("Training interrupred. Save model checkpoint? (y/n): \n")
        if i.lower().startswith("y"):
            SAVE_CHECKPOINT = True
            config["last_epoch"] = ep
            print("Saving.")
        elif i.lower().startswith("n"):
            SAVE_CHECKPOINT = False

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # model will be saved if save_checkpoint was set to True at beginning
        config["last_epoch"] = ep
        print("*ERROR* Training Interrupted.", "Checkpoint saved." if SAVE_CHECKPOINT else "")
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)


    if not LOG_WANDB:
        # Save metrics for analysis
        log = utils.JsonLog(f"./checkpoints/{MODEL_SAVE_NAME}_metricslog.json")
        log.write(train_loss=ep_losses, train_rgbdist=ep_dists, train_acc=ep_accs,
                  val_loss=val_losses, val_rgbdists=val_dists, val_acc=val_accs)
    
    # Save model and configuration
    if SAVE_CHECKPOINT:
        utils.save_model_with_shape(model, 
                                    save_dir="./checkpoints",
                                    save_name=MODEL_SAVE_NAME, 
                                    config_file=f"{MODEL_SAVE_NAME}_config.json", 
                                    config=config)
        saved_successfully = True

    