"""
Original work Copyright 2019 Davy Neven,  KU Leuven (licensed under CC BY-NC 4.0 (https://github.com/davyneven/SpatialEmbeddings/blob/master/license.txt))
Modified work Copyright 2021 Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)
Modified work Copyright 2022 Katharina Löffler, Karlsruhe Institute of Technology (MIT License)
Modifications: remove 3d parts; change lr scheduler; change visualization; train/ eval on image pairs
"""
import shutil

import torch
import os
from torch.utils.data import DataLoader


from criterions.loss import EmbedTrackLoss
from models.net import TrackERFNet
from datasets.dataset import get_dataset
from utils.logging import AverageMeter, Logger
from utils.clustering import Cluster
# from utils.visualize import VisualizeTraining
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
global_iter = 0
global_epoch = 0

scaler = torch.cuda.amp.GradScaler()


def train(logger, n_sigma, args, virtual_batch_multiplier=None,
          display=None, display_it=None, grid_x=None, grid_y=None, pixel_x=None, pixel_y=None):
    # this is without virtual batches!
    # put model into training mode
    global global_iter
    model.train()

    if virtual_batch_multiplier:
        optimizer.zero_grad()  # Reset gradients tensors

    pbar = tqdm(train_dataset_it,
                bar_format="{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]")
    # pbar.set_description(f"{'Iteration' if self.cfg.SOLVER.MAX_EPOCHS == 1 else 'Epoch'} "
    #                      f"[{self.epoch}/{self.cfg.SOLVER.MAX_EPOCHS}]")

    for i, sample in enumerate(pbar):
        global_iter += 1
        curr_frames = sample["image_curr"]  # curr frames
        prev_frames = sample["image_prev"]  # prev frames
        offset = sample["flow"].squeeze(1).to(device)  # 1YX
        seg_curr, seg_prev, tracking = model(curr_frames, prev_frames)  # B 5 Y X
        output = (torch.cat([seg_curr, seg_prev], dim=0), tracking)
        instances = torch.cat([sample["instance_curr"], sample["instance_prev"]], dim=0).squeeze(1)
        class_labels = torch.cat([sample["label_curr"], sample["label_prev"]], dim=0).squeeze(1)
        center_images = torch.cat([sample["center_image_curr"], sample["center_image_prev"]], dim=0).squeeze(1)
        loss, loss_parts = loss_fcn(output, instances, class_labels, center_images, offset,
                                    w_inst=args.W_INST, w_var=args.W_VAR, w_seed=args.W_SEED)

        if virtual_batch_multiplier:
            loss = loss / virtual_batch_multiplier  # Normalize our loss (if averaged)

        loss = loss.mean()

        if not virtual_batch_multiplier:
            optimizer.zero_grad()
            scaler.scale(loss).backward()  # Backward pass
            scaler.step(optimizer)
            scheduler.step()
        else:
            scaler.scale(loss).backward()  # Backward pass
            if (i + 1) % virtual_batch_multiplier == 0:  # Wait for several backward steps
                scaler.step(optimizer)  # Now we can do an optimizer step
                scheduler.step()
                optimizer.zero_grad()  # Reset gradients tensors

        scaler.update()

        # image_pair = (sample["image_curr"][0], sample["image_prev"][0])
        # prediction = (output[0][0], output[1][0])
        # ground_truth = (instances[0].to(device), center_images[0].to(device), offset[0].to(device))
        # prev_instance = instances[len(instances) // 2].to(device)

        # x = visualize_training(prediction, ground_truth, prev_instance, image_pair)

        if virtual_batch_multiplier:
            logger.add_scalar('train/loss', loss * virtual_batch_multiplier, global_iter)
        else:
            logger.add_scalar('train/loss', loss, global_iter)

        logger.add_scalar('train/instance_loss', loss_parts['instance'], global_iter)
        logger.add_scalar('train/variance_loss', loss_parts['variance'], global_iter)
        logger.add_scalar('train/seed_loss', loss_parts['seed'], global_iter)
        logger.add_scalar('train/track_loss', loss_parts['track'], global_iter)

        for param_group in optimizer.param_groups:
            pbar.set_postfix({'loss': loss.item(), 'lr': param_group["lr"]})
            logger.add_scalar('train/lr', param_group["lr"], global_iter)
        pbar.update(i - pbar.n)


def val(logger, n_sigma, calc_iou, args, virtual_batch_multiplier=None,
        display=None, display_it=None, grid_x=None, grid_y=None, pixel_x=None, pixel_y=None,):
    # put model into eval mode
    model.eval()
    with torch.no_grad():
        iou_score = []
        pbar = tqdm(val_dataset_it,
                    bar_format="{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]")
        for i, sample in enumerate(pbar):
            curr_frames = sample["image_curr"]  # curr frames
            prev_frames = sample["image_prev"]  # prev frames
            offset = sample["flow"].squeeze(1).to(device)  # 1YX
            seg_curr, seg_prev, tracking = model(curr_frames, prev_frames)  # B 5 Y X, B 5 Y X, B 2 Y X
            output = (torch.cat([seg_curr, seg_prev], dim=0), tracking)
            instances = torch.cat([sample["instance_curr"], sample["instance_prev"]], dim=0).squeeze(1)
            class_labels = torch.cat([sample["label_curr"], sample["label_prev"]], dim=0).squeeze(1)
            center_images = torch.cat([sample["center_image_curr"], sample["center_image_prev"]], dim=0).squeeze(1)
            if calc_iou:
                loss, loss_parts, iou_meter = loss_fcn(output, instances, class_labels, center_images, offset,
                                                       w_inst=args.W_INST, w_var=args.W_VAR, w_seed=args.W_SEED,
                                                       iou=calc_iou)
                pbar.set_postfix({'loss': loss.item(), 'iou': iou_meter.item()})
                iou_score.append(iou_meter)
            else:
                loss, loss_parts = loss_fcn(output, instances, class_labels, center_images, offset,
                                            w_inst=args.W_INST, w_var=args.W_VAR, w_seed=args.W_SEED, iou=calc_iou)
                pbar.set_postfix({'loss': loss.item()})
            loss = loss.mean()

            logger.add_scalar("val/loss", loss)
            logger.add_scalar('val/instance_loss', loss_parts['instance'])
            logger.add_scalar('val/variance_loss', loss_parts['variance'])
            logger.add_scalar('val/seed_loss', loss_parts['seed'])
            logger.add_scalar('val/track_loss', loss_parts['track'])

            pbar.update(i - pbar.n)

    if calc_iou:
        return iou_meter.mean()
    else:
        return 0.


def save_checkpoint(state, is_best, epoch, save_dir, name="checkpoint.pth"):
    print("=> saving checkpoint")
    file_name = os.path.join(save_dir, name)
    torch.save(state, file_name)
    if epoch % 10 == 0:
        file_name2 = os.path.join(save_dir, str(epoch) + "_" + name)
        torch.save(state, file_name2)
    if is_best:
        shutil.copyfile(file_name, os.path.join(save_dir, "best_iou_model.pth"))


def begin_training(cfg, set_transforms):
    if cfg.TRAIN.SAVE:
        if not os.path.exists(cfg.TRAIN.SAVE_DIR):
            os.makedirs(cfg.TRAIN.SAVE_DIR)

    if cfg.TRAIN.DISPLAY:
        plt.ion()
    else:
        plt.ioff()
        plt.switch_backend("agg")

    # define global variables
    global train_dataset_it, val_dataset_it, model, loss_fcn, optimizer, cluster, scheduler  # , visualize_training

    # train dataloader

    train_dataset = get_dataset(cfg, 'train', set_transforms)
    train_dataset_it = DataLoader(train_dataset, batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE, shuffle=True,
                                  drop_last=True, num_workers=cfg.DATALOADER.WORKERS,
                                  pin_memory=True if cfg.CUDA else False)

    # val dataloader
    val_dataset = get_dataset(cfg, 'val', set_transforms)
    val_dataset_it = DataLoader(val_dataset, batch_size=cfg.TRAIN.VAL_BATCH_SIZE, shuffle=True, drop_last=True,
                                num_workers=cfg.DATALOADER.WORKERS, pin_memory=True if device == "cuda" else False)

    # set model
    model = TrackERFNet(input_channels=cfg.MODEL.INPUT_CHANNELS,
                        n_classes=[*cfg.MODEL.N_SEG_CLASSES, cfg.MODEL.N_TRACK_CLASSES])
    model.init_output(cfg.LOSS.N_SIGMA)
    model = torch.nn.DataParallel(model).to(device)

    cluster = Cluster(cfg.TRAIN.GRID_Y, cfg.TRAIN.GRID_X, cfg.TRAIN.PIXEL_Y, cfg.TRAIN.PIXEL_X)

    cluster = cluster.to(device)
    loss = EmbedTrackLoss(grid_y=cfg.TRAIN.GRID_Y, grid_x=cfg.TRAIN.GRID_X, pixel_y=cfg.TRAIN.PIXEL_Y,
                          pixel_x=cfg.TRAIN.PIXEL_X, cluster=cluster, n_sigma=cfg.LOSS.N_SIGMA,
                          foreground_weight=cfg.LOSS.FOREGROUND_WEIGHT)

    loss_fcn = torch.nn.DataParallel(loss).to(device)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.TRAIN.LEARNING_RATE,
        total_steps=cfg.TRAIN.N_EPOCHS * len(train_dataset_it) // cfg.TRAIN.VIRTUAL_TRAIN_BATCH_MULTIPLIER,
    )

    # visualize_training = VisualizeTraining(
    #     cluster,
    #     cfg.TRAIN.SAVE_DIR,
    #     grid_x=cfg.TRAIN.GRID_X,
    #     grid_y=cfg.TRAIN.GRID_Y,
    #     pixel_x=cfg.TRAIN.PIXEL_X,
    #     pixel_y=cfg.TRAIN.PIXEL_Y,
    #     n_sigma=cfg.LOSS.N_SIGMA,
    #     device=device
    # )

    # resume
    start_epoch = 0
    best_iou = 0
    if cfg.TRAIN.RESUME_PATH is not None and os.path.exists(cfg.TRAIN.RESUME_PATH):
        print("Resuming model from {}".format(cfg.TRAIN.RESUME_PATH))
        state = torch.load(cfg.TRAIN.RESUME_PATH)
        start_epoch = state["epoch"] + 1
        best_iou = state["best_iou"]
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optim_state_dict"])
        logger = Logger(state["logger_path"])
    else:
        logger = Logger()

    for epoch in range(start_epoch, cfg.TRAIN.N_EPOCHS):
        print("Starting epoch {}".format(epoch))
        with torch.cuda.amp.autocast():
            if cfg.TRAIN.VIRTUAL_TRAIN_BATCH_MULTIPLIER > 1:
                train(
                    logger=logger,
                    virtual_batch_multiplier=cfg.TRAIN.VIRTUAL_TRAIN_BATCH_MULTIPLIER,
                    n_sigma=cfg.LOSS.N_SIGMA,
                    args=cfg.LOSS
                )
            else:
                train(
                    logger=logger,
                    display=cfg.TRAIN.DISPLAY,
                    display_it=cfg.TRAIN.DISPLAY_IT,
                    n_sigma=cfg.LOSS.N_SIGMA,
                    grid_x=cfg.TRAIN.GRID_X,
                    grid_y=cfg.TRAIN.GRID_Y,
                    pixel_x=cfg.TRAIN.PIXEL_X,
                    pixel_y=cfg.TRAIN.PIXEL_Y,
                    args=cfg.LOSS,
                )

        calc_iou = True if epoch > cfg.TRAIN.N_EPOCHS / 4 else False
        if cfg.TRAIN.VIRTUAL_VAL_BATCH_MULTIPLIER > 1:
            val_iou = val(
                logger=logger,
                virtual_batch_multiplier=cfg.TRAIN.VIRTUAL_VAL_BATCH_MULTIPLIER,
                calc_iou=calc_iou,
                n_sigma=cfg.LOSS.N_SIGMA,
                args=cfg.LOSS
            )
        else:
            val_iou = val(
                logger=logger,
                display=cfg.TRAIN.DISPLAY,
                display_it=cfg.TRAIN.DISPLAY_IT,
                n_sigma=cfg.LOSS.N_SIGMA,
                grid_x=cfg.TRAIN.GRID_X,
                grid_y=cfg.TRAIN.GRID_Y,
                pixel_x=cfg.TRAIN.PIXEL_X,
                pixel_y=cfg.TRAIN.PIXEL_Y,
                calc_iou=calc_iou,
                args=cfg.LOSS
            )

        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)

        if cfg.TRAIN.SAVE:
            state = {
                "epoch": epoch,
                "best_iou": best_iou,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
            }
        save_checkpoint(state, is_best, epoch, save_dir=cfg.TRAIN.SAVE_DIR)
