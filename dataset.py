from prompt_points import label_to_point_prompt_local, label_to_point_prompt_global
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import random
import numpy as np
from collections import *

from display import show_result_sample_figure
import albumentations as alb
from tqdm import tqdm


class octa500_2d_dataset(Dataset):
    def __init__(self, 
                 fov="3M", 
                 label_type="LargeVessel", 
                 prompt_positive_num=-1, 
                 prompt_negative_num=-1, 
                 is_local=True,
                 is_training=True):
        
        self.prompt_positive_num = prompt_positive_num
        self.prompt_negative_num = prompt_negative_num
        self.is_local = is_local
        self.is_training = is_training

        layers = ["OPL_BM", "ILM_OPL", "FULL"]
        data_dir = "datasets/OCTA-500"
        modal = "OCTA"
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

        prob = 0.3
        self.transform = alb.Compose([
            alb.RandomBrightnessContrast(p=prob),
            alb.CLAHE(p=prob), 
            # alb.SafeRotate(limit=15, p=prob),
            alb.VerticalFlip(p=prob),
            alb.HorizontalFlip(p=prob),
            alb.AdvancedBlur(p=prob),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, prompt_points, prompt_type, selected_component = self.get_sam_item(self.images[index], self.labels[index])  
        return image, prompt_points, prompt_type, selected_component, self.sample_ids[index]

    def get_sam_item(self, image, label):
        if self.is_training:
            transformed = self.transform(**{"image": image.transpose((1,2,0)), "mask": label[np.newaxis,:].transpose((1,2,0))})
            image, label = transformed["image"].transpose((2,0,1)), transformed["mask"].transpose((2,0,1))[0]
        ppn, pnn = self.prompt_positive_num, self.prompt_negative_num
        if self.is_local:
            random_max = 4
            if ppn == -1: ppn = random.randint(0, random_max)
            if pnn == -1: pnn = random.randint(int(ppn == 0), random_max)
            selected_component, prompt_points_pos, prompt_points_neg = label_to_point_prompt_local(label, ppn, pnn)
        else:
            selected_component, prompt_points_pos, prompt_points_neg = label_to_point_prompt_global(label, ppn, pnn)
        
        prompt_type = np.array([1] * len(prompt_points_pos) + [0] * len(prompt_points_neg))
        prompt_points = np.array(prompt_points_pos + prompt_points_neg)

        return image, prompt_points, prompt_type, selected_component
    
# if __name__=="__main__":
#     dataset = octa500_2d_dataset(is_local=True, prompt_positive_num=1, is_training=True)
#     for image, prompt_points, prompt_type, selected_component, sample_id in dataset:
#         print(np.max(selected_component))