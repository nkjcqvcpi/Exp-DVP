import os
import numpy as np
import cv2
from tqdm import tqdm


pic_path = 'ctc_raw_data/train/Fluo-C2DL-MSC/02_RES'
raw_path = 'ctc_raw_data/train/Fluo-C2DL-Huh7/01'


# pic_list = sorted([p for p in os.listdir(pic_path) if p.split('.')[1] == 'tif'])
# result = cv2.VideoWriter(f'{pic_path.split("/")[-2]}-{pic_path.split("/")[-1]}.mp4',
#                          cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 10, (512, 512), False)
# for img in tqdm(pic_list):
#     im = cv2.imread(os.path.join(pic_path, img), cv2.IMREAD_UNCHANGED)
#     im8 = im.astype(np.ubyte)
#     result.write(im8)
#
# result.release()


raw_list = sorted([p for p in os.listdir(raw_path) if p.split('.')[1] == 'tif'])
raw = cv2.VideoWriter(f'{raw_path.split("/")[-2]}-{raw_path.split("/")[-1]}.avi',
                      -1, 10, (512, 512), False)
for img in tqdm(raw_list):
    im = cv2.imread(os.path.join(raw_path, img), cv2.IMREAD_UNCHANGED)
    im8 = im.astype(np.ubyte)
    raw.write(im8)

raw.release()
