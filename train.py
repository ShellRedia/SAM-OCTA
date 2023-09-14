from sam_lora_image_encoder import LoRA_Sam
from segment_anything import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.utils.data import SubsetRandomSampler
from dataset import octa500_2d_dataset, octa_rose_dataset
from tqdm import tqdm
import numpy as np
from options import *
from common import *
import itertools
from statistics import *
from loss_functions import *
import os
import random
from display import *
from metrics import MetricsStatistics
from collections import *
from torch.utils.tensorboard import SummaryWriter
from augmentation import *

# 设置训练参数
parser = argparse.ArgumentParser(description='training arguments')
add_training_parser(parser)
add_octa500_2d_parser(parser)
args = parser.parse_args()
## 复制粘贴参数
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
check_interval = args.check_interval
data_dir = args.data_dir
k_fold = args.k_fold
remark = args.remark
fovs = args.fovs
modals = args.modals
projection_layers = args.projection_layers
metrics = args.metrics
label_types = args.label_types
#

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parameter_dct = vars(args)

def train(conditions):
    result_dir = "results/{}-{}".format(get_time_str(), "-".join(map(str, conditions)))
    parameter_dct["result_dir"] = result_dir
    if not os.path.exists(result_dir):os.makedirs(result_dir)
    num_samples = conditions[-1]
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    split = int(np.floor(num_samples / k_fold))
    for fold_i in range(k_fold): # 交叉验证, 划分训练集和验证集
        parameter_dct["fold_i"] = fold_i
        val_indices, train_indices = indices[fold_i * split:(fold_i + 1) * split], \
            indices[:fold_i * split] + indices[(fold_i + 1) * split:]
        train_sampler, val_sampler = [SubsetRandomSampler(x) for x in (train_indices, val_indices)]
        sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
        lora_sam = LoRA_Sam(sam, 4).cuda()
        lora_sam = DataParallel(lora_sam).to(device)
        sample_dir = result_dir + "/" + str(fold_i)
        pg = [p for p in lora_sam.parameters() if p.requires_grad]
        optimizer = optim.AdamW(pg, lr=1, weight_decay=1e-4)
        epoch_p = epochs // 5
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
            lr_lambda=lambda x: max(1e-5, lr * x / epoch_p if x <= epoch_p else lr * 0.97 ** (x - epoch_p)))
        metrics_statistics = MetricsStatistics(save_dir="{}/{}".format(parameter_dct["result_dir"], fold_i)) # 统计指标
        with open('results/parameters.txt', 'a') as parameter_file:
            for para, value in parameter_dct.items(): parameter_file.write(f"{para}: {value}\n")
            parameter_file.write("\n" * 3)
        training_loop(lora_sam, train_sampler, val_sampler, conditions, sample_dir, optimizer, scheduler, metrics_statistics, fold_i==0)
        metrics_statistics.close()
        break # for test ... ...

def training_loop(model, train_sampler, val_sampler, conditions, sample_dir, optimizer, scheduler, metrics_statistics, training_save=False):
    record_performance(model, val_sampler, conditions, sample_dir, 0, optimizer, metrics_statistics)
    fov, label_type, num_of_prompt_pos, num_of_prompt_total, local_mode, num_of_samples = conditions
    # dataset = octa500_2d_dataset(fov=fov, label_type=label_type,
    #                              num_of_prompt_pos=num_of_prompt_pos, 
    #                              num_of_prompt_total=num_of_prompt_total,
    #                              local_mode=local_mode)
    dataset = octa_rose_dataset(label_type=label_type,
                                num_of_prompt_pos=num_of_prompt_pos, 
                                num_of_prompt_total=num_of_prompt_total,
                                local_mode=local_mode)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    for epoch in tqdm(range(1, epochs+1)):
        for _, (images, original_size, prompt_points, point_labels, labels, sample_ids) in enumerate(train_loader, 0):
            images, labels = images.to(device), labels.to(device) # images.shape: torch.Size([4, 1024, 1024, 3])
            optimizer.zero_grad()
            preds = model(images, original_size, prompt_points, point_labels)
            loss = loss_calc(preds, labels, label_type)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % check_interval == 0: # 验证集表现:
            if epoch == epochs or training_save: 
                record_performance(model, val_sampler, conditions, sample_dir, epoch, optimizer, metrics_statistics)

def record_performance(model, val_sampler, conditions, sample_dir, epoch, optimizer, metrics_statistics, save_figure=True):
    val_dir = "{}/{:0>4}/".format(sample_dir, epoch)
    fov, label_type, num_of_prompt_pos, num_of_prompt_total, local_mode, num_of_samples = conditions
    metrics_statistics.metric_values["learning rate"].append(optimizer.param_groups[0]['lr'])
    for case in range(10):
        # dataset = octa500_2d_dataset(fov=fov, label_type=label_type,
        #                              num_of_prompt_pos=num_of_prompt_pos, 
        #                              num_of_prompt_total=num_of_prompt_total,
        #                              local_mode=local_mode,
        #                              random_seed=case+1)
        dataset = octa_rose_dataset(label_type=label_type,
                                     num_of_prompt_pos=num_of_prompt_pos, 
                                     num_of_prompt_total=num_of_prompt_total,
                                     local_mode=local_mode,
                                     random_seed=case+1)
        val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler)
        for _, (images, original_size, prompt_points, point_labels, labels, sample_ids) in enumerate(val_loader, 0):
            images, labels = images.to(device), labels.to(device)
            preds = model(images, original_size, prompt_points, point_labels)
            metrics_statistics.metric_values["Loss"].append(loss_calc(preds, labels, label_type).cpu().item())
            sample_id = sample_ids[0]
            preds = torch.gt(preds, 0.8).int()
            prompt_points = prompt_points / torch.tensor(images.shape[-2:]) * torch.tensor(original_size[-2:])
            val_images, val_labels, val_preds, prompt_points, point_labels = \
                [x[0].cpu().detach().numpy() for x in [images, labels, preds, prompt_points, point_labels]]
            val_label, val_pred = [torch.tensor(x).int() for x in [val_labels[0], val_preds[0]]]
            metrics_statistics.cal_epoch_metric(metrics, label_type, val_label, val_pred)
            # figure 
            if save_figure:
                sample_label_dir = "{}/{}".format(val_dir, label_type)
                if not os.path.exists(sample_label_dir): os.makedirs(sample_label_dir)
                suptitle = "{}-{}".format(sample_id, label_type)
                save_file_name = "{}/{}-{}.png".format(sample_label_dir, sample_id, case)
                val_image = cv2.resize(val_images[0], val_labels[0].shape) / 255
                save_result_sample_figure(val_image, val_labels[0], val_preds[0], 
                                        prompt_points.astype(int), point_labels.astype(int), suptitle, save_file_name)
                preds_dir = "{}/preds/{}".format(sample_dir, label_type)
                if not os.path.exists(preds_dir): os.makedirs(preds_dir)
                cv2.imwrite("{}/{}-{}.png".format(preds_dir, sample_id, case), val_preds[0] * 255)
    metrics_statistics.record_result(epoch)
            
def loss_calc(preds, labels, label_type):
    # vessel
    if label_type in ("Artery", "Vein", "LargeVessel"):
        return 0.7 * DiceLoss()(preds, labels) + 0.3 * clDiceLoss()(preds, labels)
    if label_type in ("Capillary", "FAZ"):
        return DiceLoss()(preds, labels)

# Global
def pipeline_octa500_vessel():
    conditions = [
        # ["6M", "FAZ", 0, 1, False], 
        # ["3M", "LargeVessel", 1, 1, True], 
        # ["6M", "Capillary", 0, 1, False], 
        # ["3M", "Capillary", 1, 30, False], 
        # ["6M", "Capillary", 1, 50, False], 
        # ["3M", "Capillary", 2, 60, False], 
        # ["6M", "Capillary", 2, 90, False],
        # ["3M", "Artery", 0, 1, False], 
        # ["6M", "Artery", 0, 1, False], 
        # ["3M", "Artery", 1, 20, False], 
        # ["6M", "Artery", 1, 20, False], 
        # ["3M", "Artery", 2, 40, False], 
        # ["6M", "Artery", 2, 40, False],
        # ["3M", "Vein", 0, 1, False], 
        # ["6M", "Vein", 0, 1, False], 
        # ["3M", "Vein", 1, 20, False], 
        # ["6M", "Vein", 1, 25, False], 
        # ["3M", "Vein", 2, 40, False], 
        # ["6M", "Vein", 2, 45, False],
        # ["6M", "Artery", 0, 1, True], 
        # ["6M", "Artery", 1, 1, True], 
        # ["6M", "Artery", 1, 2, True], 
        # ["6M", "Artery", 2, 3, True], 
        # ["6M", "Artery", 2, 4, True], 
        # ["ROSE", "LargeVessel", 0, 1, False], 
        # ["ROSE", "LargeVessel", 1, 30, False], 
        # ["ROSE", "LargeVessel", 2, 60, False], 
        # ["ROSE", "Capillary", 0, 1, False], 
        # ["ROSE", "Capillary", 1, 20, False], 
        ["ROSE", "Capillary", 2, 40, False],
    ]
    
    for fov, label_type, num_of_prompt_pos, num_of_prompt_total, local_mode in conditions:
        # num_of_samples = len(octa500_2d_dataset(fov=fov))
        num_of_samples = len(octa_rose_dataset())
        # parameters record:
        # parameter_dct["fov"] = fov
        parameter_dct["label_type"] = label_type
        parameter_dct["num_of_prompt_pos"] = str(num_of_prompt_pos)
        parameter_dct["num_of_prompt_total"] = str(num_of_prompt_total)
        parameter_dct["local_mode"] = str(local_mode)
        #
        train((fov, label_type, num_of_prompt_pos, num_of_prompt_total, local_mode, num_of_samples))

# Local:
def pipeline_octa_local():
    fovs = ["3M", "6M"]
    label_types = ["Artery", "Veins"] # "FAZ", "Capillary"]
    num_of_prompts_pos = list(range(3))
    num_of_prompts_neg = list(range(3))
    conditions = [fovs, label_types, num_of_prompts_pos, num_of_prompts_neg]

    for fov, label_type, num_of_prompt_pos, num_of_prompt_neg in itertools.product(*conditions):
        local_mode = True
        num_of_prompt_total = num_of_prompt_pos + num_of_prompt_neg
        if num_of_prompt_total:
            num_of_samples = len(octa500_2d_dataset(fov=fov))
            # parameters record:
            parameter_dct["fov"] = fov
            parameter_dct["label_type"] = label_type
            parameter_dct["num_of_prompt_pos"] = str(num_of_prompt_pos)
            parameter_dct["num_of_prompt_total"] = str(num_of_prompt_total)
            parameter_dct["local_mode"] = str(local_mode)
            #
            train((fov, label_type, num_of_prompt_pos, num_of_prompt_total, local_mode, num_of_samples))

# "Artery", "Vein"
if __name__=="__main__":
    pipeline_octa500_vessel()
    # pipeline_octa_local()