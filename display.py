import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm

alpha = 0.5

overlay = lambda x, y: cv2.addWeighted(x, alpha, y, 1-alpha, 0)

# 灰度图像->单通道图像, Grayscale image -> single-channel image
to_blue = lambda x: np.array([x, np.zeros_like(x), np.zeros_like(x)]).transpose((1,2,0)).astype(dtype=np.uint8)
to_red = lambda x: np.array([np.zeros_like(x), np.zeros_like(x), x]).transpose((1,2,0)).astype(dtype=np.uint8)
to_green = lambda x: np.array([np.zeros_like(x), x, np.zeros_like(x)]).transpose((1,2,0)).astype(dtype=np.uint8)
to_light_green = lambda x: np.array([np.zeros_like(x), x / 2, np.zeros_like(x)]).transpose((1,2,0)).astype(dtype=np.uint8)
to_yellow = lambda x: np.array([np.zeros_like(x), x, x]).transpose((1,2,0)).astype(dtype=np.uint8)

to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)

def show_result_sample_figure(image, label, pred, prompt_points):
    sz = image.shape[-1] // 100
    cvt_img = lambda x: x.astype(np.uint8)
    image, label, pred = map(cvt_img, (image, label, pred))
    if len(image.shape) == 2: image = to_3ch(image)
    else: image = image.transpose((1, 2, 0))
    label, pred = cv2.resize(label, image.shape[:2]), cv2.resize(pred, image.shape[:2])
    label_img = overlay(image, to_light_green(label))
    pred_img = overlay(image, to_yellow(pred))
    def draw_points(img):
        for x, y, type in prompt_points:
            cv2.circle(img, (x, y), int(1.5 * sz), (255, 0, 0), -1)
            if type: cv2.circle(img, (x, y), sz, (0, 255, 0), -1)
            else: cv2.circle(img, (x, y), sz, (0, 0, 255), -1)
    draw_points(label_img)
    draw_points(pred_img)
    return np.concatenate((image, label_img, pred_img), axis=1)

def show_prompt_points_image(image, positive_region, negative_region, positive_points, negative_points, save_file=None):
    overlay_img = overlay(to_red(negative_region), to_yellow(positive_region))
    overlay_img = overlay(to_3ch(image), overlay_img)

    for x, y in positive_points: cv2.circle(overlay_img, (x, y), 4, (0, 255, 0), -1)
    for x, y in negative_points: cv2.circle(overlay_img, (x, y), 4, (0, 0, 255), -1)

    if save_file: cv2.imwrite(save_file, overlay_img)

    return overlay_img

def view_result_samples(result_dir):
    result_dir = "results/2024-01-01-10-46-22/3M_LargeVessel_100_True/0/0020" # Your result dir
    save_dir = "sample_display/{}".format(result_dir[len("results/"):])
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    file_names = [x[-9:-4] for x in os.listdir(result_dir) if "label" in x]
    data_name = [x[:-16] for x in os.listdir(result_dir) if "label" in x][0]
    for file_name in tqdm(file_names):
        label = np.load("{}/{}_label_{}.npy".format(result_dir, data_name, file_name))
        pred = np.load("{}/{}_pred_{}.npy".format(result_dir, data_name, file_name))
        prompt_info = np.load("{}/{}_prompt_info_{}.npy".format(result_dir, data_name, file_name))
        image = np.load("{}/{}_sample_{}.npy".format(result_dir, data_name, file_name))

        result = show_result_sample_figure((1-image) * 255, label * 255, pred * 255, prompt_info)
        cv2.imwrite("{}/{}.png".format(save_dir, file_name), result)

if __name__=="__main__":
    image_dir = "datasets/OCTA-500/OCTA_3M/ProjectionMaps/OCTA(OPL_BM)"
    keypoint_dir = "datasets/OCTA-500/OCTA_3M/Keypoints"

    sample_ids = [x[:-4] for x in sorted(os.listdir(image_dir))]

    for sample_id in sample_ids:
        image = cv2.imread("{}/{}.bmp".format(image_dir, sample_id), cv2.IMREAD_GRAYSCALE)
        w, h = image.shape
        image_p = overlay(to_3ch(image), to_3ch(image))
        with open("{}/{}.txt".format(keypoint_dir, sample_id), "r") as file:
            lines = int(file.readline())
            for _ in range(lines):
                x, y = [float(t) for t in file.readline().split()]
                x, y = int(w * x), int(h * y)
                cv2.circle(image_p, (x, y), 4, (0, 255, 0), -1)
        cv2.imwrite("view_keypoints/{}.png".format(sample_id), image_p)