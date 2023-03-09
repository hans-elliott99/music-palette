#!/usr/bin/env python
import os
import sys
import time
import copy
import random
import collections
import wandb 
from dotenv import dotenv_values

import numpy as np
import torch
import torch.nn as nn
from torchaudio.transforms import FrequencyMasking, TimeMasking

import src.utils as utils
import src.dataload as dataload
from src.dataload import TransformX, TransformY, WavToSpecColorDataset, WavToSpecGTZANDataset
from src.models.transformers import SpecPatchGPT
import src.models.configs as configs


PATCH_BASED_MODELS = (SpecPatchGPT)

def get_gpt_input(X):
    """Get input for generative/autoregressive pretraining.

    :param torch.tensor X: The natural input to the model. 
        X.shape = (batch_size, num_patches, num_feats)
    :return: A tensor which has been shifted forward one patch, or time-step.
    :rtype: torch.tensor
    """
    # Add "start" patch to top of sequence (-1 tensor of shape (1,1,feats))...
    # Remove last patch in sequence...
    # Since model is using one patch to predict the next.
    batch_size, _, feats = X.shape
    start_patch = (torch.zeros(batch_size, 1, feats) - 1).to(X.device) #(b, 1, feats)
    return torch.column_stack( (start_patch, X[:, :-1, :]) )           #(b, n_patches, feats)


def patchtransformer_update(X, X_aug, y, model, criterion, config, metrics):
    """Complete one forward-pass of the patch-transformer model.
    """
    batch_size = X.shape[0]

    # Use the augmented spectrogram randomly (but not if in eval mode)
    if random.random() < config["spec_aug_p"] and model.training:
        X_in = X_aug
    else:
        X_in = X

    # FORWARD PASS    
    if config["pretraining"]:
        # predicting X using lagged X, which may be augmented.
        X_lag = get_gpt_input(X_in)                  #(b, n_patches, feats)
        logits = model(X_lag, pretraining=True)  #(b, n_patches, feats)
        loss = criterion(logits, X) #evaluate with unaugmented spec
    else:
        if config["task"] == "color":
            n_colors = int(model.n_classes / 3)
            logits_flat = model(X_in)      #(b, n_colors*3)     
            logits_mat = logits_flat.reshape(batch_size, n_colors, 3) #(b, n_colors, rgb[3])

            truth_mat = y[:, :n_colors, :]                 #(b, n_colors, 3)
            truth_flat = truth_mat.reshape(batch_size, -1) #(b, n_colors*3)
            loss = criterion(logits_flat, truth_flat)
            # for metrics, use "_mat" shaped tensors
            truths = truth_mat
            logits = logits_mat

        elif config["task"] == "genre":
            logits = model(X_in)
            loss = criterion(logits, y)
            truths = y 

    # Calc metrics
    for k in metrics.keys():
        metrics[k].step(truths, logits)
    return loss, metrics


def batched_trainloop(train_dataloader, model, criterion, metrics, optimizer,
                      config, device, batch_print_freq=10):
    """Train model using mini-batch gradient descent for one epoch.
    """
    
    model.train()
    for k in metrics.keys():
        metrics[k].reset()
    losses, batch_times = (utils.RunningMean() for _ in range(2))

    grad_accum = config["gradient_accumulation_steps"]
    n_true_batches = int( len(train_dataloader) / grad_accum )
    effective_batch = 0

    steps_since_update = 0
    time0 = time.time()
    loss = 0
    for step, (X, X_aug, y) in enumerate(train_dataloader):

        # Put data on device
        X = X.to(device)
        X_aug = X_aug.to(device)
        y = y.to(device)

        # FORWARD
        if isinstance(model, PATCH_BASED_MODELS):
            loss_, metrics = patchtransformer_update(
                X, X_aug, y, model, criterion, config, metrics
                )
            loss += loss_

        # BACKPROP AND UPDATE
        # Only if gradient_accumulation_steps have been completed
        steps_since_update += 1
        if (steps_since_update - grad_accum == 0 or
                step == len(train_dataloader)-1):

            # Backprop Update
            loss.backward()
            if config["grad_clip"] != 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), 
                                         config["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


            # Update Batch Metrics
            losses.update(loss.item() / grad_accum)
            batch_times.update(time.time() - time0)

            for k in metrics.keys():
                metrics[k].batch_update(scale=1./grad_accum) #update running mean & reset step value

            # Standard Output
            if ((effective_batch+1) % batch_print_freq == 0 or
                 effective_batch in [0, len(train_dataloader)-1]):
                print(f" [Batch {effective_batch+1}/{ n_true_batches } (et={batch_times.last :.2f}s)]")
                metric_str = ' \t'.join([
                    f"{k}={metrics[k].batch.mean :.5f}" for k in metrics.keys()
                ])
                print(f" \tLoss={losses.last :.5f} \t{metric_str}")
            
            # Reset Batch Stats
            steps_since_update = 0
            loss = 0
            time0 = time.time()

            effective_batch += 1

    return losses, metrics

@torch.no_grad()
def batched_validloop(valid_dataloader, model, criterion, 
                      metrics, config, device):
    """Evaluate the model on the validation set.
    """
    model.eval()
    losses = utils.RunningMean()
    for k in metrics.keys():
        metrics[k].reset()

    for X, X_aug, y in valid_dataloader:
        X = X.to(device)
        X_aug = X_aug.to(device)
        y = y.to(device)
        if isinstance(model, PATCH_BASED_MODELS):
            loss, metrics = patchtransformer_update(X, X_aug, y, 
                                                    model, criterion, 
                                                    config, metrics)
        losses.update(loss.item())

    for k in metrics.keys():
        metrics[k].batch_update(scale=1./len(valid_dataloader))
    return losses, metrics

@torch.no_grad()
def end_of_epoch_examples(train_dataset, model, config, device, epoch):
    model.eval()
    idx = random.randint(0, len(train_dataset)-1)
    X, X_aug, y = train_dataset[idx]
    true_mel = utils.wav_to_melspec(
        f"{train_dataset.data_dir}/{train_dataset.paths_list[idx]}")
    true_mel /= np.amax(np.abs(true_mel))

    X = X.unsqueeze(0).to(device) #(1, n_patches, feats)
    y = y.to(device)              #(1, n_colors*3)

    if config["pretraining"]:
        X_gpt = get_gpt_input(X)
        pred = model(X_gpt, pretraining=True)
        utils.plot_predicted_melspec(true_mel, 
                                     pred[0].cpu(), 
                                     savename=f"./output/pred_mel_{epoch}.jpg",
                                     figsize=(12,10))
        print(" \tNew generated MelSpec saved.")

    else:
        if config["task"] == "color":
            pred = model(X) #(1, n_colors*3)
            print(" \tExample palettes - predicted vs true:")
            print(f" \t{[round(i,1) for i in (pred[0]*255).tolist()]} | {(y[0]*255).tolist()}")
        
        elif config["task"] == "genre":
            pred = model(X) #(1, n_classes)
            pred = pred.max(1).indices.item()
            print(f" \tEx: Predicted Class = {pred} | True Class: {y.item()}")


def resolve_lr(optim, config, step):
    order = config["optim_groups"]
    if any(config['decay_lrs'].values()):
        # Update lr: pass in the current iter (1-based), relative
        # to starting iter. 
        new_lrs = []
        for i in range(len(optim.param_groups)):
            param_name = optim.param_groups[i]["name"]
            assert param_name == order[i], "resolve_lr: Optim groups out of order."

            if not config["decay_lrs"][param_name]:
                # If not decaying this optimizer group, use current lr
                new_lrs.append(optim.param_groups[i]["lr"])
            else:
                # Otherwise, update the learning rate
                lr_new = utils.lr_linearwarmup_cosinedecay(
                    iter_1_based=step + 1, 
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

def transfer_weights(oldmodel, newmodel):
    old_state_dict = oldmodel.state_dict() #pretrained model
    new_state_dict = newmodel.state_dict()

    modded_state_dict = collections.OrderedDict()
    for k in new_state_dict.keys():
        if "head" not in k:
            modded_state_dict[k] = old_state_dict[k]
        else:
            modded_state_dict[k] = new_state_dict[k]

    newmodel.load_state_dict(modded_state_dict)
    return newmodel


GTZAN_CLASSES = {"blues":0, "classical":1, "country":2, "disco":3, "hiphop":4,
                  "jazz":5, "metal":6, "pop":7, "reggae":8, "rock":9}

if __name__=="__main__":

    #NOTE:
    #Try higher weight decay for regualrization.
    #Try a much bigger model with regularizationg.
    #Also, just try implenting a SOTA model.
    # -------------------------USER CONTROL------------------------- #
    seed = 87271
    task = "genre" #genre or color
    AUDIO_DIR      = "./data/pop_videos/audio_wav" if task=="color" else "./data/gtzan/genres_clipped"
    AUDIO_COLOR_MAP= "./data/pop_videos/audio_color_mapping.csv" if task=="color" else None
    SPLITMETA_PATH = "./data/pop_videos/train_test_songs.json" if task == "color" else "./data/gtzan/train_test_songs.json"
    
    LOG_WANDB       = True
    SAVE_CHECKPOINT = True
    WANDB_RUN_GROUP = "GPTPatchTransformer_GTZAN"
    CONTINUE_RUN    = False

    MODEL_SAVE_NAME   = "gtzan_ft_1"
    MODEL_SAVE_DIR    = "./checkpoints"
    load_model_name   = "gtzan_gpt_pt" or MODEL_SAVE_NAME
    CHECKPOINT        = f"./checkpoints/{load_model_name}.pth" #will be created if doesnt exist
    CHECKPOINT_CONFIG = f"./checkpoints/{load_model_name}_config.json"

    MODEL_CLASS  = SpecPatchGPT
    MODEL_CONFIG = configs.SPGPTFinetune

    epochs = 100
    batch_size = 32 #effective batch size is _*grad_accumulation_steps
    config = {
        # Training specs
        "task"       : "genre",
        "n_colors"   : 1, #for training the color task
        "batch_size" : batch_size,
        "epochs"     : epochs,
        "pretraining" : False,
        "spec_aug"    : True, #augment spectrograms
        "spec_aug_p"  : 0.80, #percentage of time to use augmented specs

        # Utilities
        "batch_print_freq" : 30,
        "checkpoint_save_freq" : None,
        "num_workers" : 2, #for data-loading

        # Optimization
        "optim_groups" : ["embeddings", "transformers", "heads"],
        "freeze_groups": [True, False, False],
        "lrs" : {"embeddings":6e-6, "transformers":6e-5, "heads":6e-5}, #also the max lrs if decay_lr==True
        "betas" : (0.9, 0.999),
        "grad_clip" : 1.0, #disabled if 0
        "weight_decay" : 1.,
        "gradient_accumulation_steps" : 1, #to simulate larger batch-sizes

        # LR Scheduling
        "decay_lrs" : {"embeddings":True, "transformers":True, "heads":True},
        "min_lrs" : {"embeddings":6e-7, "transformers":6e-6, "heads":6e-6},
        "lr_warmup_steps" : {"embeddings":5, "transformers":5, "heads":5},
        "lr_decay_steps" : {"embeddings":epochs-5, "transformers":epochs-5, "heads":epochs-5}
    }
    
    config = { **MODEL_CONFIG().to_dict(), **config }
    if config["task"] == "color":
        metrics = {
            "rgb_acc"  : utils.Metric(fn=utils.rgb_accuracy, scale_rgb=True, window_size=8),
            "rgb_dist" : utils.Metric(fn=utils.redmean_rgb_dist, scale_rgb=True)
        }
        N_CLASSES = config["n_colors"]*3 #RGB

    elif config["task"] == "genre":
        metrics = {
            "class_acc" : utils.Metric(fn=utils.class_accuracy, is_logits=True),
            "f1_score"  : utils.Metric(fn=utils.multiclass_f1score, is_logits=True, average="macro")
        }
        N_CLASSES = len(GTZAN_CLASSES)

    if config["pretraining"]:
        metrics = {}

    y_transform = TransformY(task=config["task"], 
                             n_colors=config["n_colors"])
    
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
    if config["task"] == "color":
        filename_parser = dataload.youtube_filename_parser
        loader = WavToSpecColorDataset
        ac_map = WavToSpecColorDataset.gen_audio_color_map(AUDIO_COLOR_MAP)
        class_encoding = None
    elif config["task"] == "genre":
        filename_parser = dataload.gtzan_filename_parser
        loader = WavToSpecGTZANDataset
        ac_map = None
        class_encoding = GTZAN_CLASSES

    ## Load train/test splits, create train/valid split
    train_paths, valid_paths, test_paths = dataload.split_samples_by_song(
        audio_dir     =AUDIO_DIR,
        splitmeta_path=SPLITMETA_PATH,
        filename_parser=filename_parser,
        valid_share   =0.1,
        test_share    =0.2,
        random_seed   =seed
    )
    ## Prepare train and valid datasets
    train_dataset =     loader(train_paths,
                               audio_data_dir=AUDIO_DIR,
                               audio_color_map=ac_map,
                               class_encoding=class_encoding,
                               X_transform=X_transform,
                               y_transform=y_transform,
                               spec_augment=spec_augment
                               )

    valid_dataset =     loader(valid_paths, 
                               audio_data_dir=AUDIO_DIR,
                               audio_color_map=ac_map,
                               class_encoding=class_encoding,
                               X_transform=X_transform,
                               y_transform=y_transform
                               )

    print("Train Samples:", len(train_dataset), 
        "\nValid Samples:", len(valid_dataset),
        "\nBatch Size:", config["batch_size"],
        "| Effective Batch Size:", config["batch_size"] * config["gradient_accumulation_steps"])

    ## Create Train/Valid Dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config["batch_size"],
        shuffle = True,
        num_workers = config["num_workers"],
        drop_last = True,
        pin_memory = True if use_cuda else False
    )

    if config["gradient_accumulation_steps"] > 1:
        valid_batchsize = (config["batch_size"] - 2)*config["gradient_accumulation_steps"]
    else:
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
    if CHECKPOINT is not None and os.path.exists(CHECKPOINT):
        model_exists = True

    start_epoch = 0
    ex_x, _,_ = train_dataset[0]
    newmodel = MODEL_CLASS(X_shape=ex_x.shape, 
                        n_classes=N_CLASSES,
                        config=MODEL_CONFIG())

    if CHECKPOINT and model_exists:
        print(f"Loading model checkpoint: {CHECKPOINT}", end="... ")
        oldmodel = torch.load(CHECKPOINT, map_location=device)
        config_old = utils.load_json(CHECKPOINT_CONFIG)
        if CONTINUE_RUN:
            print("Continuing run.")
            start_epoch = config_old['last_epoch']
            config["run_id"] = config_old["run_id"]
            model = oldmodel
        else:
            print("Transferring weights.")
            model = transfer_weights(oldmodel, newmodel)
    else:
        print("Preparing new model.")
        model = newmodel

    model = model.to(device)
    print(f"Training model: {MODEL_SAVE_NAME}")
    print(f"Total params: {model.n_params :,}")

    # Freeze Parameters
    assert [g for g in model.groups.keys()] == config["optim_groups"]

    for i, group in enumerate(config["optim_groups"]):
        print(f"{group.capitalize()}:", "freezing" if config["freeze_groups"][i] else "training")
        if config["freeze_groups"][i]:
            for param in model.groups[group].parameters():
                param.requires_grad = False

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) :,}")

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
    if config["task"] == "color" or config["pretraining"]:
        criterion = nn.MSELoss()
    elif config["task"] == "genre":
        criterion = nn.CrossEntropyLoss()

    param_groups = []
    for group in config["optim_groups"]:
        param_groups.append({
            "params"  : model.groups[group].parameters(),
            "lr"      : config["lrs"][group],
            "name"    : group
        })
    optimizer = torch.optim.AdamW(
        param_groups, 
        betas=config['betas'],
        weight_decay=config['weight_decay'])

    # -------------------------TRAIN LOOP------------------------- #
    ep_losses = []
    val_losses = []
    try:
        for ep in range(start_epoch, start_epoch+config["epochs"]):
            if not torch.cuda.is_available() and use_cuda:
                raise ValueError("GPU is Lost. Exiting and saving. Reboot system.") 

            optimizer, new_lrs = resolve_lr(optimizer, config, step = ep - start_epoch)
            new_lrs = ' / '.join([f'{l :.2E}' for l in new_lrs])

            time0 = time.time()
            print(f"\nEPOCH {ep+1} / {start_epoch+config['epochs']} (embed/trfmr/head lrs={new_lrs})")
        
            train_metrics = copy.deepcopy(metrics)
            train_loss, train_metrics = batched_trainloop(
                                           train_dataloader, 
                                           model=model, 
                                           criterion=criterion,
                                           metrics=train_metrics,
                                           optimizer=optimizer,
                                           config=config, 
                                           device=device,
                                           batch_print_freq=config["batch_print_freq"])
            ep_losses.append(train_loss.mean)

            metric_str = ' \t'.join([
                f"{k}={train_metrics[k].batch.mean :.5f}" for k in train_metrics.keys()
            ])
            print(f" * Epoch Means: ",
                f"Loss={train_loss.mean :.5f} \t{metric_str}")

            valid_metrics = copy.deepcopy(metrics)
            val_loss, valid_metrics = batched_validloop(valid_dataloader,
                                           model=model, 
                                           criterion=criterion,
                                           metrics=valid_metrics,
                                           config=config,
                                           device=device)
            val_losses.append(val_loss.mean)

            metric_str = ' \t'.join([
                f"{k}={valid_metrics[k].batch.mean :.5f}" for k in valid_metrics.keys()
            ])
            print(f" * Valid Means: "
                f"Loss={val_loss.mean :.5f} \t{metric_str}")

            stp_time = time.time() - time0
            print(f" * Epoch+Valid Time: {stp_time :.1f}s (Est. Train Time ~= {stp_time * (config['epochs']) / (60*60):.2f} hrs)")

            end_of_epoch_examples(train_dataset, model, config, device, ep)


            if LOG_WANDB:
                log_metrics = {}
                for m in train_metrics.keys():
                    log_metrics[f"{m}"] = train_metrics[m].batch.mean
                for m in valid_metrics.keys():
                    log_metrics[f"val_{m}"] = valid_metrics[m].batch.mean
                wandb.log({
                    "loss"     : train_loss.mean, 
                    "val_loss" : val_loss.mean,
                    "embed_lr" : optimizer.param_groups[0]["lr"],
                    "trfmr_lr" : optimizer.param_groups[1]["lr"],
                    "head_lr"  : optimizer.param_groups[2]["lr"],
                    **log_metrics
                })

            if config["checkpoint_save_freq"] is not None and (ep+1) % config["checkpoint_save_freq"] == 0:
                print("Saving model checkpoint.")
                config["last_epoch"] = ep
                utils.save_model_with_shape(model, 
                                save_dir=MODEL_SAVE_DIR,
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
                                    save_dir=MODEL_SAVE_DIR,
                                    save_name=MODEL_SAVE_NAME, 
                                    config_file=f"{MODEL_SAVE_NAME}_config.json", 
                                    config=config)
    
    # if not LOG_WANDB:
    #     # Save metrics for analysis
    #     log = utils.JsonLog(f"./checkpoints/{MODEL_SAVE_NAME}_metricslog.json")
    #     log.write(train_loss=ep_losses, train_rgbdist=ep_dists, train_acc=ep_accs,
    #               val_loss=val_losses, val_rgbdists=val_dists, val_acc=val_accs)
