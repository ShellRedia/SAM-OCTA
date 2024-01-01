from prompt_points import label_to_point_prompt_local, label_to_point_prompt_global
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import random
import numpy as np
from collections import *
from segment_anything.utils.transforms import ResizeLongestSide

from display import show_result_sample_figure
import albumentations as alb

prob = 0.3
transform_aug = alb.Compose([
    alb.RandomBrightnessContrast(p=prob),
    alb.CLAHE(p=prob), 
    alb.Rotate(limit=30, p=prob),
    alb.VerticalFlip(p=prob),
    alb.HorizontalFlip(p=prob),
    alb.AdvancedBlur(p=prob),
])

def get_sam_item(image, label, prompt_positive_num, prompt_total_num, is_local, is_transform, model_type="vit_h"):
    if is_transform:
        transformed = transform_aug(**{"image": image.transpose((1,2,0)), "mask": label[np.newaxis,:].transpose((1,2,0))})
        image, label = transformed["image"].transpose((2,0,1)), transformed["mask"].transpose((2,0,1))[0]
    selected_component, _, prompt_points_pos, prompt_points_neg = \
        label_to_point_prompt_global(label, prompt_positive_num, prompt_total_num)
    if is_local:
        selected_component, _, prompt_points_pos, prompt_points_neg = \
            label_to_point_prompt_local(label, prompt_positive_num, prompt_total_num-prompt_positive_num)

    sam_transform = ResizeLongestSide(224) if model_type == "vit_b" else ResizeLongestSide(1024)
    original_size = tuple(image.shape[-2:])
    image = sam_transform.apply_image(image.transpose((1, 2, 0))).transpose((2, 0, 1))
    
    max_prompt_length = 50
    prompt_length = len(prompt_points_pos) + len(prompt_points_neg)
    padding_length = max_prompt_length - prompt_length

    prompt_type = np.array([1] * len(prompt_points_pos) + [0] * len(prompt_points_neg) + [2] *  padding_length)
    prompt_points = prompt_points_pos + prompt_points_neg + [[-100, -100]] * padding_length # -100 can be any constant
    prompt_points = np.array(prompt_points)
    prompt_points = sam_transform.apply_coords(prompt_points, original_size)
        
    return image, original_size, prompt_points, prompt_type, selected_component

class octa500_2d_dataset(Dataset):
    def __init__(self, data_dir="datasets/OCTA-500", 
                 fov="3M", modal="OCTA", 
                 layers=["OPL_BM", "ILM_OPL", "FULL"], 
                 label_type="LargeVessel", 
                 model_type="vit_h",
                 prompt_positive_num=-1, 
                 prompt_total_num=-1, 
                 is_local=True,
                 is_training=True):
        
        self.prompt_positive_num = prompt_positive_num
        self.prompt_total_num = prompt_total_num
        self.is_local = is_local
        self.is_training = is_training
        self.model_type = model_type

        label_dir = "{}/OCTA_{}/GT_{}".format(data_dir, fov, label_type)
        self.sample_ids = [x[:-4] for x in sorted(os.listdir(label_dir))]
        images = []
        for sample_id in self.sample_ids:
            image_channels = []
            for layer in layers:
                image_path = "{}/OCTA_{}/ProjectionMaps/{}({})/{}.bmp".format(data_dir, fov, modal, layer, sample_id)
                image_channels.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
            images.append(np.array(image_channels))
        self.images = images

        load_label = lambda sample_id: cv2.imread("{}/{}.bmp".format(label_dir, sample_id), cv2.IMREAD_GRAYSCALE) / 255
        self.labels = [load_label(x) for x in self.sample_ids]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        ppn, ptn = self.prompt_positive_num, self.prompt_total_num
        random_max = 4
        if ptn == -1: ptn = random.randint(0, random_max)
        ppn = random.randint(0, ptn) if ppn == -1 else min(ppn, ptn)

        image, original_size, prompt_points, prompt_type, selected_component = \
            get_sam_item(self.images[index], self.labels[index], ppn, ptn, self.is_local, self.is_training, self.model_type)  
        return image, original_size, prompt_points, prompt_type, selected_component, self.sample_ids[index]

class cell_dataset(Dataset):
    def __init__(self, data_dir="datasets/CELL", 
                 prompt_positive_num=-1, 
                 prompt_total_num=-1, 
                 is_local=True,
                 is_training=True):
        
        self.prompt_positive_num = prompt_positive_num
        self.prompt_total_num = prompt_total_num
        self.is_local = is_local
        self.is_training = is_training
        
        self.sample_files = sorted(os.listdir("{}/labels".format(data_dir)))
        self.sample_ids = [x[:-4] for x in self.sample_files]
        self.images = [cv2.imread("{}/images/{}.jpg".format(data_dir, x), cv2.IMREAD_COLOR) for x in self.sample_ids]
        self.images = [x.transpose((2, 0, 1)) for x in self.images]
        self.label_paths = ["{}/labels/{}.png".format(data_dir, x) for x in self.sample_ids]
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        ppn, ptn = self.prompt_positive_num, self.prompt_total_num
        if ptn == -1: ptn = random.randint(0, 4)
        ppn = random.randint(0, ptn) if ppn == -1 else min(ppn, ptn)

        image, original_size, prompt_points, prompt_type, selected_component = \
            get_sam_item(self.images[index], self.label_paths[index], ppn, ptn, self.is_local, self.is_training)  
        return image, original_size, prompt_points, prompt_type, selected_component, self.sample_ids[index]
    
if __name__=="__main__":
    pass