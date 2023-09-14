from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np

device = "cuda"

image = cv2.imread('temp/skadi.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_point = np.array([[500, 375], [213, 32], [543, 43], [453, 45]])
input_label = np.array([1, 1, 1, 1])

sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
sam.to(device=device)

predictor = SamPredictor(sam)

predictor.set_image(image)

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

for i, mask in enumerate(masks):
    cv2.imwrite("temp/{:0>4}.png".format(i), 255 * mask.astype(int))