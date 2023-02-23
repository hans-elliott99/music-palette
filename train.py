#!/usr/bin/env python
import os
import sys
import time
import random
import wandb 
from dotenv import dotenv_values

import numpy as np
import torch
import torch.nn as nn
from torchaudio.transforms import FrequencyMasking, TimeMasking

import src.utils as utils
from src.dataload import split_samples_by_song, AudioToSpecDataset, TransformX, TransformY
from src.models.transformers import PatchTransformer, GPTPatchTransformer
import src.models.configs as configs


PATCH_BASED_MODELS = (PatchTransformer, GPTPatchTransformer)

def get_gpt_input(X):
    """X.shape == (batch_size, num_patches, num_feats)
    """
    # Add "start" patch to top of sequence (-1 tensor of shape (1,1,feats))...
    # Remove last patch in sequence...
    # Since model is using one patch to predict the next.
    batch_size, _, feats = X.shape
    start_patch = (torch.zeros(batch_size, 1, feats) - 1).to(X.device) #(b, 1, feats)
    return torch.column_stack( (start_patch, X[:, :-1, :]) )           #(b, n_patches, feats)


def patchtransformer_update(X, X_aug, y, model, criterion, config):

    batch_size = X.shape[0]

    # FORWARD PASS    
    if config["pretraining"]:
        # predicting X using lagged X, which may be augmented.
        X_aug_lag = get_gpt_input(X_aug)                  #(b, n_patches, feats)
        logits = model(X_aug_lag, pretraining=True)  #(b, n_patches, feats)
        loss = criterion(logits, X)

    else:
        logits_flat = model(X) #(b, n_colors*3)     
        logits_mat = logits_flat.reshape(batch_size, model.n_colors, 3) #(b, n_colors, rgb[3])

        truth_mat = y[:, :model.n_colors, :]           #(b, n_colors, 3)
        truth_flat = truth_mat.reshape(batch_size, -1) #(b, n_colors*3)
        loss = criterion(logits_flat, truth_flat)

    # Calc metrics
    if config["pretraining"]:
        rgb_dist = rgb_acc = 0
    else:
        rgb_dist = utils.redmean_rgb_dist(logits_mat, truth_mat, scale_rgb=True).item()
        rgb_acc  = utils.rgb_accuracy(logits_mat, truth_mat, scale_rgb=True, window_size=10)

    return loss, rgb_acc, rgb_dist


def batched_trainloop(train_dataloader, model, criterion, optimizer, config, device, batch_print_freq=10):
    
    model.train()
    grad_accum     = config["gradient_accumulation_steps"]
    n_true_batches = int( len(train_dataloader) / grad_accum )
    losses, accs, dists, batch_times = (utils.RunningMean() for _ in range(4))
    effective_batch    = 0

    loss, rgb_acc, rgb_dist = 0, 0, 0
    steps_since_update = 0
    time0 = time.time()
    for step, (X, X_aug, y) in enumerate(train_dataloader):

        # Put data on device
        X = X.to(device)
        X_aug = X_aug.to(device)
        y = y.to(device)


        if isinstance(model, PATCH_BASED_MODELS):
            loss_, rgbacc_, rgbdist_ = patchtransformer_update(
                X, X_aug, y, model, criterion, config
                )
            loss     += loss_
            rgb_acc  += rgbacc_
            rgb_dist += rgbdist_

        # GRADIENT ACCUMULATION - Simulate larger batch-size by accumulating loss
        steps_since_update += 1
        if steps_since_update - grad_accum == 0 or step == len(train_dataloader)-1:

            # Backprop
            loss.backward()
            if config["grad_clip"] != 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


            # Track metrics
            losses.update(loss.item() / grad_accum)
            accs.update(rgb_acc / grad_accum)
            dists.update(rgb_dist / grad_accum)
            batch_times.update(time.time() - time0)

            # Standard Output
            if (effective_batch+1) % batch_print_freq == 0 or effective_batch in [0, len(train_dataloader)-1]:
                print(f" [Batch {effective_batch+1}/{ n_true_batches } (et={batch_times.last :.2f}s)]")
                print(f" \tLoss={losses.last :.5f} \tRGBdist={dists.last :.5f} \tAcc={accs.last :.5f}")

            # Reset
            steps_since_update = 0
            loss, rgb_acc, rgb_dist = 0, 0, 0
            time0 = time.time()
            effective_batch += 1


    return losses, dists, accs


@torch.no_grad()
def batched_validloop(valid_dataloader, model, criterion, config, device):
    
    model.eval()
    losses, accs, dists = (utils.RunningMean() for _ in range(3))
    
    for X, X_aug, y in valid_dataloader:
        X = X.to(device)
        X_aug = X_aug.to(device)
        y = y.to(device)
        if isinstance(model, PATCH_BASED_MODELS):
            loss, rgb_acc, rgb_dist = patchtransformer_update(X, X_aug, y, 
                                                            model, criterion, config)
        # Track metrics
        losses.update(loss.item())
        accs.update(rgb_acc)
        dists.update(rgb_dist)
    
    return losses, dists, accs

@torch.no_grad()
def end_of_epoch_examples(train_dataset, model, config, device, epoch):
    model.eval()
    idx = random.randint(0, len(train_dataset)-1)
    X, X_aug, y = train_dataset[idx]
    true_mel = utils.wav_to_melspec(f"{train_dataset.data_dir}/{train_dataset.paths_list[idx]}")
    true_mel /= np.amax(np.abs(true_mel))

    X = X.unsqueeze(0).to(device) #(1, n_patches, feats)
    X_aug = X_aug.unsqueeze(0).to(device)
    y = y.to(device)              #(1, n_colors*3)

    if config["pretraining"]:
        X_gpt = get_gpt_input(X_aug)
        pred = model(X_gpt, pretraining=True)
        utils.plot_predicted_melspec(true_mel, 
                                     pred[0].cpu(), 
                                     savename=f"./output/pred_mel_{epoch}.jpg",
                                     figsize=(12,10))
        print(" \tNew generated MelSpec saved.")

    elif not config["pretraining"]:
        pred = model(X) #(1, n_colors*3)
        print(" \tExample palettes - predicted vs true:")
        print(f" \t{[round(i,1) for i in (pred[0]*255).tolist()]} | {(y[0]*255).tolist()}")


def resolve_lr(optim, config, **kwargs):
    order = config["optim_groups"]
    if any(config['decay_lrs'].values()):
        print("--updating lrs--")
        # Update lr: pass in the current iter (1-based), relative
        # to starting iter. 
        new_lrs = []
        for i in range(len(optim.param_groups)):
            if not config["decay_lrs"][param_name]:
                new_lrs.append(optim.param_groups[i]["lr"])
            else:
                param_name = optim.param_groups[i]["name"]
                assert param_name == order[i]

                lr_new = utils.lr_linearwarmup_cosinedecay(
                    iter_1_based=kwargs["step"] - kwargs["start_epoch"] + 1, 
                    max_lr      =config['lrs'][param_name],
                    min_lr      =config['min_lrs'][param_name],
                    warmup_iters=config['lr_warmup_steps'][param_name],
                    decay_iters =config['lr_decay_steps'][param_name]
                    )
                optim.param_groups[i]['lr'] = lr_new
                new_lrs.append(lr_new)
    else:
        new_lrs = [v for v in config["lrs"].values()]
    return optim, new_lrs

if __name__=="__main__":

    #NOTE:
    # Generatively pretraining allows the model fine-tuned on the palette task to converge quickly.
    # However, validation performance is not improved.
    # So, focus on a more regularized pretraining (dropout, patchout, increasing the amount of data since can train in a generative fashion on unlabeled audio )
    # and a better labeled fine-tuning dataset.
    # -------------------------USER CONTROL------------------------- #
    seed = 87271
    AUDIO_DIR      = "./data/pop_videos/audio_wav"
    AUDIO_COLOR_MAP= "./data/pop_videos/audio_color_mapping.csv"
    SPLITMETA_PATH = "./data/pop_videos/train_test_songs.json"
    
    LOG_WANDB       = True
    SAVE_CHECKPOINT = True
    WANDB_RUN_GROUP = "GPTPatchTransformer"
    CONTINUE_RUN    = True

    MODEL_SAVE_NAME        = "gpt_reg"
    LOAD_CHECKPOINT        = f"./checkpoints/{MODEL_SAVE_NAME}.pth" #will be created if doesnt exist
    LOAD_CHECKPOINT_CONFIG = f"./checkpoints/{MODEL_SAVE_NAME}_config.json"

    MODEL_CLASS  = GPTPatchTransformer
    MODEL_CONFIG = configs.GPTPatchTrfmrRegConfig
    DATALOADER   = AudioToSpecDataset

    epochs = 10
    batch_size = 32 #effective batch size is _*grad_accumulation_steps
    config = {
        # Training specs
        "batch_size" : batch_size,
        "epochs"     : epochs,
        "pretraining"  : True,
        "spec_aug"  : True, #augment spectrograms
        "freeze_embeddings" : False,
        "freeze_transformer": False,
        "freeze_head"  : True,

        # Utilities
        "batch_print_freq" : 30,
        "checkpoint_save_freq" : None,
        "num_workers" : 2, #for data-loading

        # Optimizer
        "optim_groups" : ["embeddings", "transformers", "heads"],
        "lrs" : {"embeddings":3e-6, "transformers":3e-6, "heads":3e-6}, #also the max lrs if decay_lr==True
        "betas" : (0.9, 0.999),
        "grad_clip" : 1.0,     #disabled if 0
        "weight_decay" : 3e-5,
        "gradient_accumulation_steps" : 1, #to simulate larger batch-sizes

        # LR Scheduling
        "decay_lrs" : {"embeddings":False, "transformers":False, "heads":False},
        "min_lrs" : {"embeddings":3e-5, "transformers":3e-5, "heads":3e-5},
        "lr_warmup_steps" : {"embeddings":10, "transformers":10, "heads":10},
        "lr_decay_steps" : {"embeddings":epochs-10, "transformers":epochs-10, "heads":epochs-10}
    }

    config = { **MODEL_CONFIG().to_dict(), **config }

    y_transform = TransformY(n_colors=config["n_colors"])
    
    X_transform = TransformX(n_patches=config["n_patches"],
                             pad_method="mean",
                             spec_aug=config["spec_aug"],
                             flatten_patches=True)

    if config["spec_aug"]:
        spec_augment = torch.nn.Sequential(
            FrequencyMasking(freq_mask_param=16),
            TimeMasking(time_mask_param=16),
            FrequencyMasking(freq_mask_param=12),
            TimeMasking(time_mask_param=8),
        )
    else:
        spec_augment = None

    # -------------------------SETUP RUN------------------------- #
    # no need to touch these
    config["run_id"] = wandb.util.generate_id() 
    config["last_epoch"] = 0  #will be overriden by checkpoint if one is loaded

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    torch.cuda.empty_cache()

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print(time.strftime(r"%Y-%m-%d %H:%M"))
    print("CUDA:", use_cuda)

    # -------------------------PREPARE DATA------------------------- #
    train_paths, valid_paths, test_paths = split_samples_by_song(
        audio_dir     =AUDIO_DIR,
        splitmeta_path=SPLITMETA_PATH,
        valid_share   =0.1,
        test_share    =0.2,
        random_seed   =seed
    )
    ac_map = AudioToSpecDataset.gen_audio_color_map(AUDIO_COLOR_MAP)

    train_dataset = DATALOADER(train_paths,
                               audio_data_dir=AUDIO_DIR,
                               audio_color_map=ac_map,
                               X_transform=X_transform,
                               y_transform=y_transform,
                               spec_augment=spec_augment
                               )

    valid_dataset = DATALOADER(valid_paths, 
                               audio_data_dir=AUDIO_DIR,
                               audio_color_map=ac_map,
                               X_transform=X_transform,
                               y_transform=y_transform)

    print("Train Samples:", len(train_dataset), 
        "\nValid Samples:", len(valid_dataset),
        "\nBatch Size:", config["batch_size"],
        "| Effective Batch Size:", config["batch_size"] * config["gradient_accumulation_steps"])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config["batch_size"],
        shuffle = True,
        num_workers = config["num_workers"],
        drop_last = True,
        pin_memory = True if use_cuda else False
    )

    valid_batchsize = (config["batch_size"] - 2)*config["gradient_accumulation_steps"]
    if config["gradient_accumulation_steps"] == 1:
        valid_batchsize = config["batch_size"]

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = valid_batchsize,
        shuffle = True,
        num_workers = config["num_workers"],
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
        ex_x, _,_ = train_dataset[0]
        model = MODEL_CLASS(X_shape=ex_x.shape, config=MODEL_CONFIG())

    else:
        print(f"Loading model checkpoint: {LOAD_CHECKPOINT}")
        model = torch.load(LOAD_CHECKPOINT)
        config_old = utils.load_json(LOAD_CHECKPOINT_CONFIG)
        if CONTINUE_RUN:
            start_epoch = config_old['last_epoch']
            config["run_id"] = config_old["run_id"]

    model = model.to(device)
    print(f"Model params: {model.n_params :,}")

    print("Freezing" if config["freeze_embeddings"] else "Training", "embedding layers.")
    print("Freezing" if config["freeze_transformer"] else "Training", "transformer layers.")
    print("Freezing" if config["freeze_head"] else "Training", "head/classification layers.")

    for name, child in model.named_children():
        if ("ln_0" in name or "linear_proj" in name or "embed" in name) and config["freeze_embeddings"]:
            for param in child.parameters():
                param.requires_grad = False
        if ("transformer" in name) and config["freeze_transformer"]: 
            for param in child.parameters():
                param.requires_grad = False
        if ("head" in name) and config["freeze_head"]:
            for param in child.parameters():
                param.requires_grad = False

    # Change dropout percentages if the config calls for modified values
    model = utils.update_dropout_p(model, config, verbose=True)

    # ----------------------------WANDB----------------------------- #
    wandb_key = dotenv_values(".env")["WANDB_KEY"]
    if LOG_WANDB:
        wandb.login(key=wandb_key)
        wandb.init(
            id=config["run_id"],
            resume="allow",
            project="MusicPalette",
            config=config,
            group=WANDB_RUN_GROUP,
            anonymous="allow"
        )
        wandb.watch(model)

    # -------------------------LOSS & OPTIM------------------------- #
    criterion = nn.MSELoss()
    
    assert [n for n in model.groups.keys()] == config["optim_groups"]

    param_groups = []
    for name in config["optim_groups"]:
        param_groups.append({
            "params"  : model.groups[name].parameters(),
            "lr"      : config["lrs"][name],
            "name"    : name
        })
    optimizer = torch.optim.AdamW(
        param_groups, 
        betas=config['betas'],
        weight_decay=config['weight_decay'])

    # -------------------------TRAIN LOOP------------------------- #
    ep_losses, ep_dists, ep_accs = [], [], []
    val_losses, val_dists, val_accs = [], [], []
    try:
        for ep in range(start_epoch, start_epoch+config["epochs"]):

            optimizer, new_lrs = resolve_lr(optimizer, config, step=ep, start_epoch=start_epoch)
            new_lrs = ' / '.join([f'{l :.2E}' for l in new_lrs])

            time0 = time.time()
            print(f"\nEPOCH {ep+1} / {start_epoch+config['epochs']} (embed/trfmr/head lrs={new_lrs})")
            
            el, ed, ea = batched_trainloop(train_dataloader, 
                                           model=model, 
                                           criterion=criterion, 
                                           optimizer=optimizer,
                                           config=config, 
                                           device=device,
                                           batch_print_freq=config["batch_print_freq"])
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
            print(f" * Epoch+Valid Time: {stp_time :.1f}s (Est. Train Time ~= {stp_time * (config['epochs']) / (60*60):.2f} hrs)")

            end_of_epoch_examples(train_dataset, model, config, device, ep)


            if LOG_WANDB:
                wandb.log({
                    "loss"     : el.mean, "val_loss"     : vl.mean,
                    "rgb_dist" : ed.mean, "val_rgb_dist" : vd.mean,
                    "acc"      : ea.mean, "val_acc"      : va.mean,
                    "embed_lr" : optimizer.param_groups[0]["lr"],
                    "trfmr_lr" : optimizer.param_groups[1]["lr"],
                    "head_lr"  : optimizer.param_groups[2]["lr"]
                })

            if config["checkpoint_save_freq"] is not None and (ep+1) % config["checkpoint_save_freq"] == 0:
                print("Saving model checkpoint.")
                config["last_epoch"] = ep
                utils.save_model_with_shape(model, 
                                save_dir="./checkpoints",
                                save_name=f"{MODEL_SAVE_NAME}_ep{ep}", 
                                config_file=f"{MODEL_SAVE_NAME}_config.json", 
                                config=config)
            # -*-*-end of epoch iters-*-*-
        config["last_epoch"] = ep

    except KeyboardInterrupt as e:
        inp = input("Training interrupred. Save model checkpoint? (y/n): \n")
        if inp.lower().startswith("y"):
            SAVE_CHECKPOINT = True
            config["last_epoch"] = ep
            print("Saving.")
        elif inp.lower().startswith("n"):
            SAVE_CHECKPOINT = False

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

        MODEL_SAVE_NAME = f"{MODEL_SAVE_NAME}_es"
        print("*ERROR* Unexpected Interruption. Saving emergency checkpoint: ", MODEL_SAVE_NAME)
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        SAVE_CHECKPOINT = True
        config["last_epoch"] = ep
    

    if SAVE_CHECKPOINT:
        # Save model and configuration
        utils.save_model_with_shape(model, 
                                    save_dir="./checkpoints",
                                    save_name=MODEL_SAVE_NAME, 
                                    config_file=f"{MODEL_SAVE_NAME}_config.json", 
                                    config=config)
    
    # if not LOG_WANDB:
    #     # Save metrics for analysis
    #     log = utils.JsonLog(f"./checkpoints/{MODEL_SAVE_NAME}_metricslog.json")
    #     log.write(train_loss=ep_losses, train_rgbdist=ep_dists, train_acc=ep_accs,
    #               val_loss=val_losses, val_rgbdists=val_dists, val_acc=val_accs)
