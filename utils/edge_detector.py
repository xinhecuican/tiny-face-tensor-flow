import math
import time

import numpy as np
import cv2
from matplotlib import pyplot as plt


def sobel(img: np.ndarray, threshold, bboxes=None):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # shape = gray_img.shape
    # ans = np.zeros(shape)
    x = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    dst = dst.astype(np.float32)
    # if bboxes is not None:
    #     for box in bboxes:
    #         ymin = int(box[0]) if int(box[0]) >= 0 else 0
    #         ymax = int(box[2]) if int(box[2]) <= 500 else 500
    #         xmin = int(box[1]) if int(box[1]) >= 0 else 0
    #         xmax = int(box[3]) if int(box[3]) <= 500 else 500
    #         for i in range(xmin, xmax):
    #             for k in range(ymin, ymax):
    #                 if dst[i, k] > threshold:
    #                     dst[i, k] = 255
    return dst
