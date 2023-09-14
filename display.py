import matplotlib.pyplot as plt
import numpy as np
import cv2

def save_result_sample_figure(image, label, pred, prompt_points, point_labels, suptitle, save_file_name, overlay=True):
    if overlay:
        alpha = 0.5
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        def gray2yellow(gray_image):
            rnt = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.float32)
            rnt[:, :, 0] = gray_image
            rnt[:, :, 1] = gray_image
            return rnt
        label, pred = [gray2yellow(x) for x in [label, pred]]
        label, pred = [cv2.addWeighted(image.astype(np.float32), 1 - alpha, x.astype(np.float32), alpha, 0) for x in [label, pred]]
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    plt.suptitle(suptitle)
    for i, (subimage, subtitle) in enumerate(zip([image, label, pred], ["input_image", "ground_truth", "prediction"])):
        plt.subplot(1, 3, i+1)
        if len(subimage.shape) == 2: subimage = np.repeat(subimage[:, :, np.newaxis], 3, axis=2)
        for (x, y), t in zip(prompt_points, point_labels):
            if t:
                cv2.circle(subimage, (x, y), 4, (0, 1, 0), -1) # draw the prompt points
            else:
                cv2.circle(subimage, (x, y), 4, (1, 0, 0), -1)
        plt.imshow(subimage)
        plt.axis('off')
        plt.title(subtitle)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(save_file_name)
    plt.close(fig)