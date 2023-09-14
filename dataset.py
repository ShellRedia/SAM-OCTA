from convert import points_to_gaussian_heatmap, label_to_point_prompt
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np
from collections import *
from segment_anything.utils.transforms import ResizeLongestSide

def get_sam_item(image, label_path, num_of_prompt_pos, num_of_prompt_total, local_mode, random_seed):
    transform = ResizeLongestSide(1024)
    original_size = tuple(image.shape[-2:])
    image = transform.apply_image(image.transpose((1, 2, 0)))
    image = image.transpose((2, 0, 1))
    component, prompt_points_pos, prompt_points_neg = label_to_point_prompt(label_path, 
                                                        positive_num=num_of_prompt_pos, 
                                                        total_num=num_of_prompt_total, 
                                                        local_mode=local_mode,
                                                        random_seed=random_seed)
    prompt_points, prompt_label = None, None
    if len(prompt_points_pos) and len(prompt_points_neg):
        prompt_points = np.vstack((prompt_points_pos, prompt_points_neg))
    elif len(prompt_points_pos):
        prompt_points = prompt_points_pos
    elif len(prompt_points_neg):
        prompt_points = prompt_points_neg

    if prompt_points is not None:
        prompt_label = np.array([1] * len(prompt_points_pos) + [0] * len(prompt_points_neg))
        prompt_points = transform.apply_coords(prompt_points, original_size)

    return image, original_size, prompt_points, prompt_label, component

class octa500_2d_dataset(Dataset):
    def __init__(self, data_dir="datasets/OCTA-500", 
                 fov="3M", modal="OCTA", 
                 projection_layers=["OPL_BM", "ILM_OPL", "FULL"],
                 selected=["all"], label_type="LargeVessel", 
                 num_of_prompt_pos=5, num_of_prompt_total=5, 
                 local_mode=False, random_seed=None):
        self.image_size = 304 if fov == "3M" else 400
        self.label_excel_path = "{}/OCTA_{}/TextLabels.xlsx".format(data_dir, fov)
        
        self.random_seed = random_seed
        self.num_of_prompt_pos = num_of_prompt_pos
        self.num_of_prompt_total = num_of_prompt_total
        self.local_mode = local_mode
        # 读取样本信息，通过条件筛选出 ID
        valid_ids = self.get_valid_ids(selected)
        self.ids = valid_ids
        # 获取筛选样本
        images = []
        for id in valid_ids:
            image_channels = []
            for layer_name in projection_layers:
                for modal_type in "OCTA", "OCT":
                    if modal_type == modal or modal.upper() == "ALL":
                        image_path = "{}/OCTA_{}/ProjectionMaps/{}({})/{}.bmp".format(data_dir, fov, modal_type, layer_name, id)
                        image_channels.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
            images.append(np.array(image_channels))
        self.images = images
        # 获取标签
        self.label_paths = ["{}/OCTA_{}/GT_{}/{}.bmp".format(data_dir, fov, label_type, id) for id in valid_ids]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, original_size, prompt_points, prompt_label, component = get_sam_item(self.images[index], 
                                                                                    self.label_paths[index], 
                                                                                    self.num_of_prompt_pos, 
                                                                                    self.num_of_prompt_total, 
                                                                                    self.local_mode, self.random_seed)
        
        return image, original_size, prompt_points, prompt_label, component, self.ids[index]

    def get_valid_ids(self, selected):
        label_excel_path = self.label_excel_path
        ids = pd.read_excel(label_excel_path)["ID"]
        diseases = pd.read_excel(label_excel_path)["Disease"]
        return [id for id, disease in zip(ids, diseases) if disease in selected or "all" in selected]

class octa_rose_dataset(Dataset):
    def __init__(self, data_dir="datasets/OCTA-ROSE", 
                 label_type="LargeVessel",
                 num_of_prompt_pos=5, num_of_prompt_total=5, 
                 local_mode=False, random_seed=None):
        
        self.random_seed = random_seed
        self.num_of_prompt_pos = num_of_prompt_pos
        self.num_of_prompt_total = num_of_prompt_total
        self.local_mode = local_mode
        
        label_type = "GT_" + label_type
        images, self.label_paths, self.ids = [], [], []
        for img_file in os.listdir("/".join([data_dir, label_type])):
            img_channels = []
            for layer in "DVC", "IVC", "SVC":
                img_path = "/".join([data_dir, "ProjectionMaps", layer, img_file])
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_channels.append(img)
            self.ids.append(img_file[:-4])
            self.label_paths.append("/".join([data_dir, label_type, img_file]))
            images.append(img_channels)
        self.images = np.array(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, original_size, prompt_points, prompt_label, component = get_sam_item(self.images[index], 
                                                                                    self.label_paths[index], 
                                                                                    self.num_of_prompt_pos, 
                                                                                    self.num_of_prompt_total, 
                                                                                    self.local_mode, self.random_seed)
        return image, original_size, prompt_points, prompt_label, component, self.ids[index]

class octa_ss_dataset(Dataset):
    def __init__(self, data_dir="datasets/OCTA-SS",
                 num_of_prompt_pos=5, num_of_prompt_total=5, 
                 local_mode=False, random_seed=None):
        
        self.random_seed = random_seed
        self.num_of_prompt_pos = num_of_prompt_pos
        self.num_of_prompt_total = num_of_prompt_total
        self.local_mode = local_mode

        images, self.label_paths, self.ids = [], [], []
        for img_file in os.listdir("/".join([data_dir, "segmented_images"])):
            gray_image = cv2.imread("/".join([data_dir, "original_images", img_file]), cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            images.append(np.transpose(image, (2, 0, 1)))
            self.ids.append(img_file[:-4])
            self.label_paths.append("/".join([data_dir, "segmented_images", img_file]))
        self.images = np.array(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, original_size, prompt_points, prompt_label, component = get_sam_item(self.images[index], 
                                                                                    self.label_paths[index], 
                                                                                    self.num_of_prompt_pos, 
                                                                                    self.num_of_prompt_total, 
                                                                                    self.local_mode, self.random_seed)
        return image, original_size, prompt_points, prompt_label, component, self.ids[index]
    

# if __name__=="__main__":
#     dataset = octa500_2d_dataset(random_seed=114)
#     print(dataset[0][0].shape, np.max(dataset[0][0]))
#     dataset = octa_rose_dataset(random_seed=114)
#     print(dataset[0][0].shape, np.max(dataset[0][0]))
#     dataset = octa_ss_dataset(random_seed=114)
#     print(dataset[0][0].shape, np.max(dataset[0][0]))