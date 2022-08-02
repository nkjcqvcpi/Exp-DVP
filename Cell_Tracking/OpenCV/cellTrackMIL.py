import os
import re

import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm


class CellTrack:
    raw_data = './ctc_raw_data'

    def __init__(self, dataset, threshold, mode='train'):
        self.threshold = threshold
        self.data_path = os.path.join(self.raw_data, mode, dataset)

    def __call__(self, series):
        self.res_path = os.path.join(self.data_path, series + '_RES')
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)
        self.res_track = pd.DataFrame(columns=["t_start", "t_end", "predecessor"])
        self.img_path = os.path.join(self.data_path, series)
        self.img_list = sorted([i for i in os.listdir(self.img_path) if i[-4:] == '.tif'])
        trackers, oks, bboxs = self.init_track()
        self.tracking(trackers, oks, bboxs)

    def segmentation(self, im):
        gray = cv.imread(os.path.join(self.img_path, im), cv.IMREAD_UNCHANGED)
        img = cv.imread(os.path.join(self.img_path, im))
        ret, thresh = cv.threshold(gray, self.threshold, 255, cv.THRESH_BINARY)  # + cv.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.15 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        retval, markers, stats, centroids = cv.connectedComponentsWithStats(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        markers = cv.watershed(img, markers)
        return stats[1:], markers

    def init_track(self):
        bboxs, _ = self.segmentation(self.img_list[0])
        bboxs[:, 4] = np.arange(1, len(bboxs) + 1)
        trackers = {i[4]: cv.TrackerMIL_create() for i in bboxs}
        frame0 = cv.imread(os.path.join(self.img_path, self.img_list[0]), cv.IMREAD_UNCHANGED)
        oks = {tracker: trackers[tracker].init(frame0, bboxs[tracker - 1][:-1]) for tracker in trackers.keys()}
        print(2)
        for o in oks.keys():
            self.res_track.loc[o] = [0, len(self.img_list) - 1, 0]
        print(3)
        bboxs = {i[4]: np.zeros(4) for i in bboxs}
        return trackers, oks, bboxs

    def tracking(self, trackers, oks, bboxs):
        for f, img in enumerate(tqdm(self.img_list)):
            frame = cv.imread(os.path.join(self.img_path, img), cv.IMREAD_UNCHANGED)
            bs, result = self.segmentation(img)
            result = np.clip(result, 1, 2**16) - 1
            result = np.uint16(result)
            for tracker in trackers.keys():
                ok, bbox = trackers[tracker].update(frame)
                bboxs[tracker] = bbox
                oks[tracker] = ok
                if not ok:
                    self.res_track.loc[tracker, 't_end'] = f - 1
            cv.imwrite(os.path.join(self.res_path, f'mask{re.search("[0-9]+", img).group()}.tif'), result)

        self.res_track.to_csv(os.path.join(self.res_path, 'res_track.txt'), sep=' ', header=False)


if __name__ == '__main__':
    ct = CellTrack('Fluo-C2DL-Huh7', 20)  # 7
    ct('02')
    i = 0
