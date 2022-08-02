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
        self.raw = cv.VideoWriter(f'{series}_RAW.avi', cv.VideoWriter_fourcc(*'MJPG'), 5, (1024, 1024), False)
        self.res = cv.VideoWriter(f'{series}_RES.avi', cv.VideoWriter_fourcc(*'MJPG'), 5, (1024, 1024), False)
        self.res_path = os.path.join(self.data_path, series + '_RES')
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)
        self.res_track = pd.DataFrame(columns=["t_start", "t_end", "predecessor"])
        self.img_path = os.path.join(self.data_path, series)
        self.img_list = sorted([i for i in os.listdir(self.img_path) if i[-4:] == '.tif'])
        trackers, oks = self.init_track()
        self.tracking(trackers)
        self.raw.release()
        self.res.release()

    def segmentation(self, im):
        gray = cv.imread(os.path.join(self.img_path, im), cv.IMREAD_UNCHANGED)
        img = cv.imread(os.path.join(self.img_path, im))
        ret, thresh = cv.threshold(gray, self.threshold[0], 255, cv.THRESH_BINARY)  # + cv.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv.dilate(opening, kernel, iterations=3)
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, self.threshold[1] * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        retval, markers, stats, centroids = cv.connectedComponentsWithStats(sure_fg, 4)
        markers += 1
        markers[unknown == 255] = 0
        markers = cv.watershed(img, markers)
        markers = np.uint16(np.clip(markers, 1, 2 ** 16) - 1)
        for b, stat in enumerate(stats):
            if stat[4] < self.threshold[2]:
                markers[markers == b] = 0
                stats[b, :] = 0
        stats[:, 4] = np.arange(len(stats))
        stats = stats[stats.sum(axis=1) != 0, :]
        return stats[1:], centroids[1:], markers

    def init_track(self):
        bboxs, cid, _ = self.segmentation(self.img_list[0])
        trackers = {bbox[4]: cv.TrackerCSRT_create() for bbox in bboxs}
        frame0 = cv.imread(os.path.join(self.img_path, self.img_list[0]), cv.IMREAD_UNCHANGED)
        oks = {bbox[-1]: trackers[bbox[-1]].init(frame0, bbox[:-1]) for bbox in bboxs}
        for bbox in bboxs:
            self.res_track.loc[bbox[4]] = [0, len(self.img_list) - 1, 0]
        return trackers, oks

    def tracking(self, trackers):
        for f, img in enumerate(tqdm(self.img_list)):
            frame = cv.imread(os.path.join(self.img_path, img), cv.IMREAD_UNCHANGED)
            self.raw.write(np.uint8(frame))
            bs, cid, result = self.segmentation(img)
            for tid, tracker in trackers.items():
                ok, bbox = tracker.update(frame)
                if not ok:
                    self.res_track.loc[tid, 't_end'] = f - 1
            cv.imwrite(os.path.join(self.res_path, f'mask{re.search("[0-9]+", img).group()}.tif'), result)
            self.res.write(np.uint8(result))
        self.res_track.to_csv(os.path.join(self.res_path, 'res_track.txt'), sep=' ', header=False)


if __name__ == '__main__':
    ct = CellTrack('Fluo-C2DL-Huh7', [20, 0.05, 5])  # [20, 0.05, 5]  # [10, 0.01, 7]
    ct('02')
