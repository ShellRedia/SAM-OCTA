import numpy as np
from scipy.stats import multivariate_normal
from scipy import ndimage
import cv2
from collections import *
import random
from itertools import *
from functools import *
import os

def points_to_gaussian_heatmap(centers, height, width, scale): # 将二维的坐标点转换为高斯热图
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

@lru_cache
def point_prompt_cache_helper(label_path, local_mode=True, neg_range=20, area_limit=200):
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) / 255
    h, w = label.shape
    structure = ndimage.generate_binary_structure(2, 2)
    labeled_image, num_features = ndimage.label(label, structure=structure)
    label_cnts = Counter(labeled_image.flatten().tolist())
    filtered_labels = [k for k, v in label_cnts.items() if v >= area_limit and k]
    component = np.zeros((h, w), dtype=int)

    # if 'local' mode: choose a label
    if local_mode:
        choice_label = random.choice(filtered_labels)
        positive_points = np.argwhere(labeled_image==choice_label)
        component[labeled_image == choice_label] = 1
    else:
        positive_points = np.argwhere(labeled_image > 0)
        component[labeled_image > 0] = 1
    
    # dfs get adjacent negtive points:
    dq = deque([(x, y, 0) for x, y in positive_points])
    neg_s = set((x, y) for x, y in positive_points)
    while dq:
        x, y, d = dq.popleft()
        if d >= neg_range: continue
        for dx, dy in product(*[range(-1, 2)] * 2):
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and (nx, ny) not in neg_s:
                neg_s.add((nx, ny))
                dq.append((nx, ny, d+1))
    negative_points = list(neg_s - set((x, y) for x, y in positive_points))

    # group 
    positive_points_group = [positive_points] if local_mode \
        else [np.argwhere(labeled_image == k) for k in filtered_labels]

    return component, positive_points_group, negative_points

def label_to_point_prompt(label_path, positive_num=5, total_num=5, local_mode=True,
                          neg_range=20, area_limit=50, random_seed=None):
    if random_seed: 
        random.seed(random_seed)
        np.random.seed(random_seed)

    component, positive_points_group, negative_points = \
        point_prompt_cache_helper(label_path, local_mode, neg_range, area_limit)
    prompt_points_pos, prompt_points_neg = [], []

    # positive points:
    for positive_points in positive_points_group:
        for _ in range(positive_num):
            i = np.random.choice(np.arange(len(positive_points)))
            y, x = positive_points[i] # pick up a positive point randomly
            prompt_points_pos.append([x, y])
    # negtive points:
    for _ in range(total_num-len(prompt_points_pos)):
        i = np.random.choice(np.arange(len(negative_points)))
        y, x = negative_points[i] # pick up a negative point randomly
        prompt_points_neg.append([x, y])

    return np.array([component], dtype=float), np.array(prompt_points_pos), np.array(prompt_points_neg)

if __name__=="__main__":
    mx = 0
    dir_path = "datasets/OCTA-500/OCTA_6M/GT_Artery"
    for label_path in os.listdir(dir_path):
        t = 0
        component, pos, neg = label_to_point_prompt("datasets/OCTA-500/OCTA_6M/GT_Artery/" + label_path, 1, 30, local_mode=False)
        t += len(pos)
        component, pos, neg = label_to_point_prompt("datasets/OCTA-500/OCTA_6M/GT_Vein/" + label_path, 1, 30, local_mode=False)
        t += len(pos)
        mx = max(mx, t)
    print("mx:", mx)
    # label_path = "datasets/OCTA-500/OCTA_3M/GT_Vein/10301.bmp"
    # label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    # ori_img = cv2.imread("datasets/OCTA-500/OCTA_3M/ProjectionMaps/OCTA(ILM_OPL)/10301.bmp", cv2.IMREAD_GRAYSCALE)
    # for i in range(20):
    #     component, pos, neg = label_to_point_prompt(label_path, 3, 6)
    #     component = (255 * component[0]).astype(np.uint8)
    #     component = cv2.cvtColor(component, cv2.COLOR_GRAY2RGB)
    #     img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2RGB)
    #     label = cv2.cvtColor(label_img, cv2.COLOR_GRAY2RGB)
    #     cv2.imwrite("temp/complabel_{:0>4}.png".format(i), component)
    #     img = img * 0.4 + (0, 255, 255) * component * 0.2
    #     sz = 7
    #     for (x, y) in pos: cv2.circle(component, (x, y), sz, (0, 0, 255), -1)
    #     for (x, y) in neg: cv2.circle(component, (x, y), sz, (255, 0, 0), -1)
    #     cv2.imwrite("temp/comp_{:0>4}.png".format(i), component)
    #     for (x, y) in pos: cv2.circle(img, (x, y), sz, (0, 0, 255), -1)
    #     for (x, y) in neg: cv2.circle(img, (x, y), sz, (255, 0, 0), -1)
    #     cv2.imwrite("temp/img_{:0>4}.png".format(i), img)
    #     for (x, y) in pos: cv2.circle(label, (x, y), sz, (0, 0, 255), -1)
    #     for (x, y) in neg: cv2.circle(label, (x, y), sz, (255, 0, 0), -1)
    #     pt = 20
    #     try:
    #         for j, (x, y) in enumerate(pos): cv2.imwrite("temp/patch_{:0>4}_{:0>2}.png".format(i, j), label[y-pt:y+pt,x-pt:x+pt])
    #         for j, (x, y) in enumerate(neg): cv2.imwrite("temp/patch_{:0>4}_{:0>2}.png".format(i, j+len(pos)), label[y-pt:y+pt,x-pt:x+pt])
    #     except:
    #         pass
    #     cv2.imwrite("temp/label_{:0>4}.png".format(i), label)