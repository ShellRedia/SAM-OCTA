import numpy as np
from scipy.stats import multivariate_normal
from scipy import ndimage
import cv2
from collections import *
import random
from itertools import *
from functools import *
import os
from tqdm import tqdm
from display import show_prompt_points_image

random_seed = 0

if random_seed:  
    random.seed(random_seed)
    np.random.seed(random_seed)

# 将二维的坐标点转换为高斯热图, Converting 2D coordinate points to Gaussian heat maps
def points_to_gaussian_heatmap(centers, height, width, scale): 
    gaussians = []
    for y, x in centers:
        s = np.eye(2) * scale
        g = multivariate_normal(mean=(x, y), cov=s)
        gaussians.append(g)
    x, y = np.arange(0, width), np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.stack([xx.ravel(), yy.ravel()]).T
    zz = sum(g.pdf(xxyy) for g in gaussians)
    img = zz.reshape((height, width))

    return img / np.max(img)

def get_labelmap(label):
    structure = ndimage.generate_binary_structure(2, 2)
    labelmaps, conneted_num = ndimage.label(label, structure=structure)
    # 像素->联通分量，0为背景, Pixel->connected component, 0 is the background
    pixel2connetedId = {(x, y): val for (x, y), val in np.ndenumerate(labelmaps)}
    return labelmaps, conneted_num, pixel2connetedId

def get_negative_region(labelmap, neg_range=8):
    kernel = np.ones((neg_range, neg_range), np.uint8)
    negative_region = cv2.dilate(labelmap, kernel, iterations=1) - labelmap
    return negative_region

def label_to_point_prompt_global(label, positive_num=2, total_num=50):
    labelmaps, conneted_num, _ = get_labelmap(label)
    positive_points, negative_points = [], []
    
    connected_points_dct = defaultdict(list)
    for (x, y), val in np.ndenumerate(labelmaps): connected_points_dct[val].append((y, x))

    # time consuming loop
    for connected_id in range(1, conneted_num+1):
        if positive_num <= len(connected_points_dct[connected_id]): 
            positive_points += random.sample(connected_points_dct[connected_id], max(0, positive_num))

    negative_num = total_num - conneted_num * positive_num
    negative_region = get_negative_region(label)
    negative_points = [(y, x) for (x, y), val in np.ndenumerate(negative_region) if val]
    negative_points = random.sample(negative_points, max(0, negative_num))

    return np.array([label / 255], dtype=float), np.array([negative_region], dtype=float), positive_points, negative_points

def label_to_point_prompt_local(label, positive_num=2, negative_num=2):
    labelmaps, _, pixel2connetedId = get_labelmap(label)
    labelmap_points = [(x, y) for (x, y), val in np.ndenumerate(labelmaps) if val]

    min_area = positive_num + negative_num

    def get_selected_points():
        selected_pixel = random.randint(0, len(labelmap_points)-1)
        selected_id = pixel2connetedId[labelmap_points[selected_pixel]]
        return  [(x, y) for (x, y), val in np.ndenumerate(labelmaps) if val == selected_id]
    
    selected_points = get_selected_points()
    while len(selected_points) < min_area: selected_points = get_selected_points()
    
    selected_labelmap = np.zeros_like(labelmaps, dtype=np.uint8)
    for (x, y) in selected_points: selected_labelmap[(x, y)] = 1

    negative_region = get_negative_region(selected_labelmap)

    positive_points = [(y, x) for (x, y), val in np.ndenumerate(selected_labelmap) if val]
    negative_points = [(y, x) for (x, y), val in np.ndenumerate(negative_region) if val]

    positive_points = random.sample(positive_points, max(0, positive_num))
    negative_points = random.sample(negative_points, max(0, negative_num))

    # no prompt points, no segmentation
    if not positive_points + negative_points: selected_labelmap = np.zeros_like(labelmaps, dtype=np.uint8) 

    return np.array([selected_labelmap], dtype=float), np.array([negative_region], dtype=float), positive_points, negative_points

# if __name__=="__main__":
#     image_path = "datasets/OCTA-500/OCTA_3M/ProjectionMaps/OCTA(OPL_BM)/10301.bmp"
#     label_path = "datasets/OCTA-500/OCTA_3M/GT_LargeVessel/10301.bmp"
#     image, label = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

#     for i in tqdm(range(200)):
#         label_to_point_prompt_global(label, 2, 50)

#     selected_label, neg_region, pos, neg = label_to_point_prompt_global(label)
#     show_prompt_points_image(image, selected_label[0] * 255, neg_region[0] * 255, pos, neg, "prompt.png")
    # '''
    # 200 samples
    # label_to_point_prompt_local: 21s
    # label_to_point_prompt_global: before -> 4m 47s; now -> 15s
    # '''