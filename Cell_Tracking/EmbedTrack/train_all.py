"""
Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
Licensed under MIT License
"""
import torch.cuda

from train.run_training_pipeline import copy_ctc_data, generate_image_crops, init_training
from config import get_cfg_defaults
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', metavar='D', type=str, nargs='+')
    args = parser.parse_args()

    for data_set in args.datasets:
        cfg = get_cfg_defaults()
        cfg.merge_from_file(f"configs/{data_set}.yml")
        opts = ["TRAIN.SAVE_MODEL_PATH",
                os.path.join(cfg.TRAIN.SAVE_MODEL_PATH, data_set, "adam_norm_onecycle_" + str(cfg.TRAIN.N_EPOCHS)),
                "DATALOADER.DATA_DIR", os.path.join(cfg.DATA.DATA_PATH, cfg.DATA.DATA_SET)]

        cfg.merge_from_list(opts)

        """
Fluo-N2DH-SIM+
Fluo-C2DL-MSC
Fluo-N2DH-GOWT1
OOM PhC-C2DL-PSC 
error BF-C2DL-HSC
OOM Fluo-N2DL-HeLa
pre-error BF-C2DL-MuSC
trained DIC-C2DH-HeLa
train 
        """
        # cfg.freeze()

        copy_ctc_data(cfg.DATA)  # copy data
        generate_image_crops(cfg)  # generate crops
        init_training(cfg)  # training
        if cfg.CUDA:
            torch.cuda.empty_cache()
