"""
Original work Copyright 2022 Katharina Löffler, Karlsruhe Institute of Technology (MIT License)
parts of generate_image_crops and init_training based on code of Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)

"""
import json
from datetime import datetime
from glob import glob
import torch
from utils.transforms import get_transform, DEFAULT_TRANSFORM

from PIL import Image
from tqdm import tqdm

from datasets.generate_crops import *
from datasets.prepare_data import prepare_ctc_data
from datasets.prepare_data import prepare_ctc_gt
from train.train import begin_training
import pandas as pd
import glob

IMAGE_FORMATS = ("bmp", "jpeg", "tif", "png", "tiff")
MODES_TO_BITS = {
    "1": 1,
    "L": 8,
    "P": 8,
    "RGB": 8,
    "RGBA": 8,
    "CMYK": 8,
    "YCbCr": 8,
    "I": 32,
    "F": 32,
    "I;16": 16,
    "I;16B": 16,
    "I;16N": 16,
    "BGR;15": 15,
    "BGR;24": 24,
}


def copy_ctc_data(data_config):
    """
    Copy CTC data according to the data_config
    """
    if os.path.exists(os.path.join(data_config.DATA_PATH, data_config.DATA_SET)):
        print(f"{os.path.join(data_config.DATA_PATH, data_config.DATA_SET)} already exists, therefore no data is "
              f"copied from {os.path.join(data_config.RAW_DATA_PATH, data_config.DATA_SET)}")
    else:
        print(f"prepare data of {data_config.DATA_SET}")
        prepare_ctc_data(os.path.join(data_config.RAW_DATA_PATH, data_config.DATA_SET), data_config.DATA_PATH,
                         keep_st=data_config.USE_SILVER_TRUTH, val_split=data_config.TRAIN_VAL_SPLIT,
                         sub_dir_names=data_config.TRAIN_VAL_SEQUENCES)
        prepare_ctc_gt(os.path.join(data_config.RAW_DATA_PATH, data_config.DATA_SET), data_config.DATA_PATH,
                       val_split=data_config.TRAIN_VAL_SPLIT, sub_dir_names=data_config.TRAIN_VAL_SEQUENCES)
        print(f"data stored in {data_config.DATA_PATH}")


def generate_image_crops(config):
    """
    Generate image crops for training and evaluation
    """
    crops_dir = os.path.join(config.DATA.DATA_PATH, "crops")

    if os.path.exists(os.path.join(crops_dir, config.DATA.DATA_SET)):
        print(f"{os.path.join(crops_dir, config.DATA.DATA_SET)} already exists, therefore no crops are generated from "
              f"{os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET)}")
    else:
        img_sequences = list(os.listdir(os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, "train")))
        d_path = os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, "train", img_sequences[0], "images")
        img_files = (os.path.join(d_path, element) for element in os.listdir(d_path) if element.endswith(IMAGE_FORMATS))
        # decode pixel depth and image dimension
        with Image.open(next(img_files)) as img:
            pix_depth = MODES_TO_BITS[img.mode]
        data_sub_sets = [
            "/".join([data_split, img_sequence])
            for data_split in ["train", "val"]
            for img_sequence in os.listdir(os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, data_split))
        ]
        data_properties_dir = dict()
        if isinstance(config.TRAIN.CROP_SIZE, int):
            grid_shape = (config.TRAIN.CROP_SIZE, config.TRAIN.CROP_SIZE)

        elif len(config.TRAIN.crop_size) == 2:
            grid_shape = (config.TRAIN.CROP_SIZE[0], config.TRAIN.CROP_SIZE[1])
        else:
            raise AssertionError(f"Unknown crop size {config.TRAIN.CROP_SIZE}")

        data_properties_dir["n_y"], data_properties_dir["n_x"] = grid_shape
        data_properties_dir["data_type"] = str(pix_depth) + "-bit"

        with open(os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, "data_properties.json"), "w") as outfile:
            json.dump(data_properties_dir, outfile)
            print(f"Dataset properies of the `{config.DATA.DATA_SET}` dataset is saved to "
                  f"{os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, 'data_properties.json')}")

        print(crops_dir)
        for data_subset in data_sub_sets:
            image_dir = os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, data_subset, "images")
            instance_dir = os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, data_subset, "masks")
            image_names = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
            instance_names = sorted(glob.glob(os.path.join(instance_dir, "*.tif")))
            for i in tqdm(np.arange(len(image_names))):
                # create img crops only if segmentation mask exists
                try:
                    idx_segm_mask = instance_names.index(os.path.join(instance_dir, os.path.basename(image_names[i])))
                except ValueError:
                    continue
                # generate crops for an image
                process(image_names[i], instance_names[idx_segm_mask], os.path.join(crops_dir, config.DATA.DATA_SET),
                        data_subset, config.TRAIN.CROP_SIZE, config.TRAIN.CENTER)
            print("Cropping of images, instances and centre_images for data_subset = `{}` done!".format(data_subset))
        # generate offset images (shift between two cell centers between successive frames) for tracking
        for data_subset in data_sub_sets:
            instance_dir = os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, data_subset, "masks")
            center_image_path = os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, data_subset,
                                             f"center-{config.TRAIN.CENTER}")
            lineage_file = os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, data_subset, "lineage.txt")
            lineage_data = pd.read_csv(lineage_file, delimiter=" ", header=None,
                                       names=["cell_id", "t_start", "t_end", "predecessor"])
            lineage = {cell["cell_id"]: cell["predecessor"] for _, cell in lineage_data.iterrows()}

            calc_obj_shifts(instance_dir, center_image_path, os.path.join(crops_dir, config.DATA.DATA_SET, data_subset),
                            lineage, config.TRAIN.CROP_SIZE)
            # remove empty crops
            offset_path = os.path.join(os.path.join(crops_dir, config.DATA.DATA_SET, data_subset),
                                       (Path(center_image_path).name + "-" + "flow"), )

            centroid_path = os.path.join(os.path.join(crops_dir, config.DATA.DATA_SET, data_subset),
                                         Path(center_image_path).name)
            image_path = os.path.join(crops_dir, config.DATA.DATA_SET, data_subset, "images")
            instances_path = os.path.join(crops_dir, config.DATA.DATA_SET, data_subset, "masks")
            data_files, img_pairs = get_image_pairs(image_path, instances_path, offset_path, centroid_path)
            remove_empty_crops(*data_files, img_pairs)


def get_image_pairs(img_path, instance_path, offset_path, centroid_path):
    """
    Get all pairs of succssive frames.
    Args:
        img_path (string):path to the raw image files
        instance_path (string) : path to the segmentation mask files
        offset_path (string): path to the offset image files
        centroid_path (string): path to the centroid image files

    Returns: (tuple, list): tuple of lists containing all paths to the different image files; list of pairs of successive frames

    """
    pairs = []
    image_files, instance_files, centroid_files = list(zip(*[(
        os.path.join(img_path, img_file),
        os.path.join(instance_path, img_file),
        os.path.join(centroid_path, img_file),
    ) for img_file in os.listdir(img_path)]))
    offset_files = [os.path.join(offset_path, file) for file in os.listdir(offset_path)]
    for i, path_img_file in enumerate(image_files):
        path_img, name_img = os.path.split(path_img_file)
        name_img, ending = name_img.split(".")
        time, patch_id = name_img.split("_")
        name_next_img = ("_".join([str(int(time) + 1).zfill(len(time)), patch_id]) + "." + ending)
        flow_img_name = ("_".join([str(int(time) + 1).zfill(len(time)), time, patch_id]) + "." + ending)
        path_flow_img = os.path.join(offset_path, flow_img_name)
        try:
            pairs.append((image_files.index(os.path.join(path_img, name_next_img)),
                          i, offset_files.index(path_flow_img),))
        except ValueError:
            continue
    return (image_files, instance_files, offset_files, centroid_files), pairs


def remove_empty_crops(image_files, instance_files, offset_files, centroid_files, img_pairs):
    """
    Remove empty image crops from the train/val data set.
    Args:
        image_files (list): list of raw image files
        instance_files (list):list of mask image files
        offset_files (list): list of offset image files
        centroid_files (list): list of centroid image files
        img_pairs (list): list of successive image frames

    Returns:

    """

    for pair_idx in img_pairs:
        img_idx_curr, img_idx_prev, offset_idx = pair_idx
        try:
            instances_curr = tifffile.imread(instance_files[img_idx_curr])
        except FileNotFoundError:
            instances_curr = np.array([0])
        try:
            instances_prev = tifffile.imread(instance_files[img_idx_prev])
        except FileNotFoundError:
            instances_prev = np.array([0])

        mask_ids = np.unique(np.concatenate([instances_curr.reshape(-1), instances_prev.reshape(-1)]))
        mask_ids = mask_ids[mask_ids != 0]
        if len(mask_ids) == 0:
            delete_image_crop(instance_files[img_idx_curr])
            delete_image_crop(image_files[img_idx_curr])
            delete_image_crop(centroid_files[img_idx_curr])
            delete_image_crop(instance_files[img_idx_prev])
            delete_image_crop(image_files[img_idx_prev])
            delete_image_crop(centroid_files[img_idx_prev])
            delete_image_crop(offset_files[offset_idx])


def delete_image_crop(img_file):
    try:
        os.remove(img_file)
    except FileNotFoundError:
        pass


def init_training(config):
    """
    Initialize training of the model.
    """
    crops_dir = os.path.join(config.DATA.DATA_PATH, "crops")
    if not os.path.exists(os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, "data_properties.json")):
        data_properties_file = os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, "data_properties.json")
        raise AssertionError(f"No such file f{data_properties_file}")
    with open(os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, "data_properties.json")) as json_file:
        data = json.load(json_file)
        data_type, n_y, n_x = (data["data_type"], int(data["n_y"]), int(data["n_x"]),)

    train_subsets = [
        "/".join(["train", img_sequence])
        for img_sequence in os.listdir(os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, "train"))
        if img_sequence in config.DATA.TRAIN_VAL_SEQUENCES
    ]
    val_subsets = [
        "/".join(["val", img_sequence])
        for img_sequence in os.listdir(os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, "val"))
        if img_sequence in config.DATA.TRAIN_VAL_SEQUENCES
    ]

    if config.DATALOADER.NAME == "2d":
        set_transforms = get_transform(DEFAULT_TRANSFORM)
    else:
        raise AssertionError(f"Unknown dimensions {config.DATALOADER.NAME}")

    dataset_opts = ['DATALOADER.CROP_DIR', os.path.join(crops_dir, config.DATA.DATA_SET),
                    'DATALOADER.TRAIN_SUBSETS', train_subsets, 'DATALOADER.VAL_SUBSETS', val_subsets]

    if config.TRAIN.RESUME_TRAINING:
        resume_path = (Path(config.TRAIN.SAVE_MODEL_PATH) / "best_iou_model.pth").as_posix()
        save_dir = config.TRAIN.SAVE_MODEL_PATH
    else:
        resume_path = ""
        save_dir = os.path.join(config.TRAIN.SAVE_MODEL_PATH, datetime.now().strftime("%Y-%m-%d---%H-%M-%S"))

    min_mask_size = calc_min_mask_size([os.path.join(config.DATA.DATA_PATH, config.DATA.DATA_SET, train_p)
                                        for train_p in train_subsets])

    opts = ['TRAIN.RESUME_PATH', resume_path, 'TRAIN.SAVE_DIR', save_dir, 'TRAIN.GRID_Y', n_y, 'TRAIN.GRID_X', n_x,
            'TRAIN.MIN_MASK_SIZE', float(min_mask_size)]
    config.merge_from_list(dataset_opts)
    config.merge_from_list(opts)
    config.freeze()
    print(config)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "config.yml"), "w") as yml_file:
        yml_file.write(config.dump(allow_unicode=True))

    begin_training(config, set_transforms)


def calc_min_mask_size(train_paths):
    """Calculate the 1% percentile of the mask sizes from the train data set."""
    all_mask_sizes = []
    for train_p in train_paths:
        for file in os.listdir(os.path.join(train_p, "masks")):
            if file.endswith(".tif"):
                segm_mask = tifffile.imread(os.path.join(train_p, "masks", file))
                masks = get_indices_pandas(segm_mask)
                if len(masks) > 0:
                    mask_sizes = masks.apply(lambda x: len(x[0])).values
                    all_mask_sizes.extend(mask_sizes)
    return np.percentile(all_mask_sizes, 1)
