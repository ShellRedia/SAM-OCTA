from sam_lora_image_encoder import LoRA_Sam
from segment_anything import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.utils.data import SubsetRandomSampler
from dataset import octa500_2d_dataset
from tqdm import tqdm
import numpy as np
from options import *
import itertools
from statistics import *
from loss_functions import *
import os
import random
import time
from display import *
from metrics import MetricsStatistics
from collections import *
from segment_anything.utils.transforms import ResizeLongestSide

parser = argparse.ArgumentParser()
add_training_parser(parser)
add_octa500_2d_parser(parser)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_gpus = torch.cuda.device_count()
    
for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    print(f"GPU {i}: {gpu_name}")

time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
print(time_str)

test_weight_path = '3M_LargeVessel_Global.pth'

to_cuda = lambda x: x.to(torch.float).to(device)

ppn, pnn = args.prompt_positive_num, args.prompt_negative_num
dataset_params = [args.fov, args.label_type, ppn, pnn, args.is_local, False]
dataset_test = octa500_2d_dataset(*dataset_params)

parameters = [args.fov, args.label_type, args.epochs, args.is_local, args.model_type, args.remark]

save_dir = "test/{}/{}".format(time_str, "_".join(map(str, parameters)))

sample_n = len(dataset_test)

if args.model_type == "vit_h":
    sam = sam_model_registry["vit_h"](checkpoint="sam_weights/sam_vit_h_4b8939.pth")
elif args.model_type == "vit_l":
    sam = sam_model_registry["vit_l"](checkpoint="sam_weights/sam_vit_l_0b3195.pth")
else:
    sam = sam_model_registry["vit_b"](checkpoint="sam_weights/sam_vit_b_01ec64.pth")

sam_transform = ResizeLongestSide(224) if args.model_type == "vit_b" else ResizeLongestSide(1024)

model = LoRA_Sam(sam, 4).cuda()
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(test_weight_path))
model = torch.nn.DataParallel(model).to(device)
model.eval()

val_loader = DataLoader(dataset_test, batch_size=1)

metrics_statistics = MetricsStatistics(save_dir=save_dir)

def make_prompts(images, prompt_points):
    original_size = tuple(images.shape[-2:])
    images = sam_transform.apply_image_torch(images)
    prompt_points = sam_transform.apply_coords_torch(prompt_points, original_size)

    return images, original_size, prompt_points

with torch.no_grad():
    for images, prompt_points, prompt_type, selected_components, sample_ids in tqdm(val_loader):
        images, labels, prompt_type = map(to_cuda, (images, selected_components, prompt_type))
        images, original_size, prompt_points = make_prompts(images, prompt_points)
        preds = model(images, original_size, prompt_points, prompt_type)

        preds = torch.gt(preds, 0.8).int()
        sample_id = str(sample_ids[0])

        image, label, pred = map(lambda x:x[0][0].cpu().detach(), (images, labels, preds))
        prompt_points, prompt_type = prompt_points[0].cpu().detach(), prompt_type[0].cpu().detach()
        prompt_info = np.concatenate((prompt_points, prompt_type[:,np.newaxis]), axis=1).astype(int)
        metrics_statistics.cal_epoch_metric(args.metrics, args.label_type, label.int(), pred.int())
        
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        save_sample_func = lambda x, y: np.save("/".join([save_dir,\
                            "{}_{}_{}.npy".format(args.label_type, x, sample_id)]), y)
        save_items = {"sample":image / 255, "label":label, "prompt_info":prompt_info, "pred":pred}
        for x, y in save_items.items(): save_sample_func(x, y)


metrics_statistics.record_result(-1)
metrics_statistics.close()