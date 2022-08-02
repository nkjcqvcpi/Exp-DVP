# my_project/config.py
from yacs.config import CfgNode as CN
from pathlib import Path
import torch
import os

PROJECT_PATH = os.path.join(*Path(__file__).parts[:-1])

_C = CN()
_C.DATA = CN()
# Path where the CTC datasets are stored
_C.DATA.RAW_DATA_PATH = os.path.join(PROJECT_PATH, "ctc_raw_data/train")
# Name of the VAL_LOADER
_C.DATA.DATA_SET = ''
# Path where to store to the prepared data for training the model
_C.DATA.DATA_PATH = os.path.join(PROJECT_PATH, "data")
# Use the ST from the cell tracking challenge or use the GT annotations
_C.DATA.USE_SILVER_TRUTH = False
# list of the image sequences to use for training and validation
_C.DATA.TRAIN_VAL_SEQUENCES = ["01", "02"]
# fraction of images to split from each image sequence provided in train_val_sequences for validation
_C.DATA.TRAIN_VAL_SPLIT = 0.2

_C.MODEL = CN()
# number of input channels of the raw images
_C.MODEL.INPUT_CHANNELS = 1
# number of output channels of the two segmentation decoders
_C.MODEL.N_SEG_CLASSES = [4, 1]
# number of output channels of the tracking decoder (2D images: 2, 3D images: 3)
_C.MODEL.N_TRACK_CLASSES = 2

_C.TRAIN = CN()
# path where to store the trained models
_C.TRAIN.SAVE_MODEL_PATH = os.path.join(PROJECT_PATH, "ckpts")
# size of the quadratic image crops the model is trained on
_C.TRAIN.CROP_SIZE = 256
# indicates how the cell center is calculated (options: "centroid", "approximate-medoid", "medoid")
_C.TRAIN.CENTER = "medoid"
# continue training from a checkpoint
_C.TRAIN.RESUME_TRAINING = False
# size of the VAL_LOADER to train on if None - the full VAL_LOADER is used per epoch if an int is provided per epoch only a
# subset of the training data is used
_C.TRAIN.TRAIN_SIZE = 3000
# batch size during training
_C.TRAIN.TRAIN_BATCH_SIZE = 16
# increase batch size virtually by applying and optim step after every N batches
# (the loss is averages to loss/N per batch)
_C.TRAIN.VIRTUAL_TRAIN_BATCH_MULTIPLIER = 1
# size of the VAL_LOADER to val on if None - the full VAL_LOADER is used per epoch if an int is provided per epoch only a
# subset of the validation data is used
_C.TRAIN.VAL_SIZE = 2600
# batch size during training
_C.TRAIN.VAL_BATCH_SIZE = 16
# increase batch size virtually by averaging the loss to loss/N per batch
_C.TRAIN.VIRTUAL_VAL_BATCH_MULTIPLIER = 1
# max number of epochs to train the model
_C.TRAIN.N_EPOCHS = 200
# display visualization of the training
_C.TRAIN.DISPLAY = True
# display visualization every N iterations if display is True
_C.TRAIN.DISPLAY_IT = 50
# learning rate of the optimizer
_C.TRAIN.LEARNING_RATE = 5e-4

_C.TRAIN.RESUME_PATH = ''
_C.TRAIN.SAVE_DIR = ''
_C.TRAIN.GRID_Y = 0
_C.TRAIN.GRID_X = 0
_C.TRAIN.SAVE = True
_C.TRAIN.PIXEL_X = 1
_C.TRAIN.PIXEL_Y = 1
_C.TRAIN.MIN_MASK_SIZE = 0.

_C.LOSS = CN()
_C.LOSS.N_SIGMA = 2
_C.LOSS.FOREGROUND_WEIGHT = 1
_C.LOSS.W_INST = 1
_C.LOSS.W_VAR = 10
_C.LOSS.W_SEED = 1

_C.DATALOADER = CN()
_C.DATALOADER.NAME = '2d'
_C.DATALOADER.CENTER = "center-" + _C.TRAIN.CENTER
_C.DATALOADER.DATA_DIR = ''
_C.DATALOADER.CROP_DIR = ''
_C.DATALOADER.WORKERS = 16
_C.DATALOADER.TRAIN_SUBSETS = []
_C.DATALOADER.VAL_SUBSETS = []


_C.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_C.CUDA = _C.DEVICE == 'cuda'


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
