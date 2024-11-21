import cv2
import numpy as np
from scipy.ndimage import label, center_of_mass

# system
import os, random, time, GPUtil
from tqdm import tqdm
from collections import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# torch
import torch
import torch.optim as optim
from torch.nn import DataParallel

# SAM
from segment_anything import *
from sam_lora_image_encoder import LoRA_Sam
from segment_anything.utils.transforms import ResizeLongestSide

class DisplayManager:
    def __init__(self, save_dir="display"):
        alpha = 0.5
        self.overlay = lambda x, y: cv2.addWeighted(x, alpha, y, 1-alpha, 0)
        self.to_3ch = lambda x: np.array([x,x,x]).transpose((1,2,0)).astype(dtype=np.uint8)
        self.to_color = lambda x, color: (self.to_3ch(x) * color).astype(dtype=np.uint8)
        self.to_visible = lambda x : (x * 255 if x.max() <= 1 else x).astype(np.uint8)

        self.point_color_dct = {True:(0, 255, 0), False:(0, 0, 255)}
        self.point_size = 10
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def display_prompt(self, image, mask, prompts, save_name="temp"):
        image = self.overlay(image, self.to_color(mask, (0, 1, 1)))
        
        for x, y, z in prompts:
            cv2.circle(image, (int(x), int(y)), self.point_size, self.point_color_dct[z], -1)
        cv2.imwrite("{}/{}.png".format(self.save_dir, save_name), image)
    
    def display_predict(self, image, label, pred, prompt_points, prompt_type, save_name="temp"):
        to_numpy = lambda x : x.numpy()
        image, label, pred, prompt_points, prompt_type = map(to_numpy, [image, label, pred, prompt_points, prompt_type])
        image, label, pred = map(self.to_visible, [image, label, pred])

        image = self.to_3ch(image)
        
        image_pred = self.overlay(image, self.to_color(pred, (0, 1, 1)))
        image_pred_prompt = image_pred.copy()

        for (x, y), z in zip(prompt_points, prompt_type):
            cv2.circle(image_pred_prompt, (int(x), int(y)), self.point_size, self.point_color_dct[z], -1)

        merged_image = np.concatenate([image, image_pred_prompt], axis=1)

        cv2.imwrite("{}/{}.png".format(self.save_dir, save_name), merged_image)


class PredictManager_OCTA:
    def __init__(self, weight_path, save_dir="predition", model_type="vit_l"):
        self.device_ids = "0"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        self.to_cuda = lambda x: x.to(torch.float).to(self.device)
        
        if model_type == "vit_h": sam = sam_model_registry["vit_h"](checkpoint="sam_weights/sam_vit_h_4b8939.pth")
        elif model_type == "vit_l": sam = sam_model_registry["vit_l"](checkpoint="sam_weights/sam_vit_l_0b3195.pth")
        else: sam = sam_model_registry["vit_b"](checkpoint="sam_weights/sam_vit_b_01ec64.pth")

        self.sam_transform = ResizeLongestSide(224) if model_type == "vit_b" else ResizeLongestSide(1024)

        rank = 4
        lora_sam = LoRA_Sam(sam, rank).cuda()

        

        self.model = DataParallel(lora_sam).to(self.device)

        self.model.load_state_dict(torch.load(weight_path))
        


    def predict(self, image, save_name):
        dm = DisplayManager(save_dir=self.save_dir)

        # process image:
        w = image.shape[1]
        image, prompt_image = image[:, :w//2], image[:, w//2:]

        ppn, pnn = self.get_red_and_green_points(prompt_image)

        prompt_points = torch.tensor(np.array([ppn + pnn]))
        prompt_type = torch.tensor(np.array([[1] * len(ppn) + [0] * len(pnn)]))

        images = torch.tensor(np.array([image.transpose((2,0,1))]))

        with torch.no_grad():
            images, prompt_type = map(self.to_cuda, (images, prompt_type))
            images, original_size, prompt_points = self.make_prompts(images, prompt_points)

            preds = self.model(images, original_size, prompt_points, prompt_type)

            preds = torch.gt(preds, 0.8).int()

            image = images[0][1].cpu().detach()
            pred = preds[0][0].cpu().detach()

            prompt_points, prompt_types = prompt_points[0].cpu().detach().int(), prompt_type[0].cpu().detach().int()

            dm.display_predict(image, torch.zeros_like(pred), pred, prompt_points, prompt_types, save_name)

    def get_red_and_green_points(self, image):
  
        diff_map = np.sum(np.abs(image - (0,255,0)), axis=2)
        green_mask = np.where(diff_map < 10, 1, 0).astype(np.uint8)

        diff_map = np.sum(np.abs(image - (0,0,255)), axis=2)
        red_mask = np.where(diff_map < 10, 1, 0).astype(np.uint8)

        red_coords, green_coords = [], []

        labeled_array, num_features = label(green_mask, structure=np.ones((3,3)))

        for i in range(1, num_features + 1): 
            center = center_of_mass(green_mask, labeled_array, index=i)
            green_coords.append([int(center[1]), int(center[0])])
        
        labeled_array, num_features = label(red_mask, structure=np.ones((3,3)))
        
        for i in range(1, num_features + 1): 
            center = center_of_mass(red_mask, labeled_array, index=i)
            red_coords.append([int(center[1]), int(center[0])])

        if len(green_coords) + len(red_coords) == 0: red_coords = [[-100, -100]]

        return green_coords, red_coords

        
    def make_prompts(self, images, prompt_points):
        original_size = tuple(images.shape[-2:])
        images = self.sam_transform.apply_image_torch(images)
        prompt_points = self.sam_transform.apply_coords_torch(prompt_points, original_size)

        return images, original_size, prompt_points

if __name__=="__main__":
    weight_path = "vit_l_rv.pth"
    sample_id = 10453
    image_path = "{}.png".format(sample_id)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    pm = PredictManager_OCTA(weight_path)
    pm.predict(image, str(sample_id))
