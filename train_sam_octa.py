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

parser = argparse.ArgumentParser(description='training arguments')
add_training_parser(parser)
add_octa500_2d_parser(parser)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_gpus = torch.cuda.device_count()
    
for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    print(f"GPU {i}: {gpu_name}")

time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
print(time_str)

to_cuda = lambda x: x.to(torch.float).to(device)

class TrainManager_OCTA:
    def __init__(self, dataset_train, dataset_val):
        self.dataset_train, self.dataset_val = dataset_train, dataset_val
        parameters = [args.fov, args.label_type, args.epochs, args.is_local, args.model_type, args.remark]
        self.record_dir = "results/{}/{}".format(time_str, "_".join(map(str, parameters)))
        self.cpt_dir = "{}/checkpoints".format(self.record_dir)

        if not os.path.exists(self.cpt_dir): os.makedirs(self.cpt_dir)

        sample_n = len(self.dataset_train)
        self.indices = list(range(sample_n))
        random.shuffle(self.indices)
        self.split = sample_n // args.k_fold

        if args.model_type == "vit_h":
            sam = sam_model_registry["vit_h"](checkpoint="sam_weights/sam_vit_h_4b8939.pth")
        elif args.model_type == "vit_l":
            sam = sam_model_registry["vit_l"](checkpoint="sam_weights/sam_vit_l_0b3195.pth")
        else:
            sam = sam_model_registry["vit_b"](checkpoint="sam_weights/sam_vit_b_01ec64.pth")

        self.sam_transform = ResizeLongestSide(224) if args.model_type == "vit_b" else ResizeLongestSide(1024)

        lora_sam = LoRA_Sam(sam, 4).cuda()
        self.model = DataParallel(lora_sam).to(device)
        torch.save(self.model.state_dict(), '{}/init.pth'.format(self.cpt_dir))

        self.loss_func = DiceLoss() 
        if args.label_type in ["Artery", "Vein", "LargeVessel"]: 
            self.loss_func = lambda x, y: 0.8 * DiceLoss()(x, y) + 0.2 * clDiceLoss()(x, y)
    
    def get_dataloader(self, fold_i):
        train_indices = self.indices[:fold_i * self.split] + self.indices[(fold_i + 1) * self.split:]
        val_indices = self.indices[fold_i * self.split:(fold_i + 1) * self.split]
        train_sampler, val_sampler = [SubsetRandomSampler(x) for x in (train_indices, val_indices)]
        batch_size = len(args.device.split(","))
        train_loader = DataLoader(self.dataset_train, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(self.dataset_val, batch_size=1, sampler=val_sampler)
        
        return train_loader, val_loader

    def reset(self):
        self.model.load_state_dict(torch.load('{}/init.pth'.format(self.cpt_dir)))
        pg = [p for p in self.model.parameters() if p.requires_grad] # lora parameters
        self.optimizer = optim.AdamW(pg, lr=1, weight_decay=1e-4)
        epoch_p = args.epochs // 5
        lr_lambda = lambda x: max(1e-5, args.lr * x / epoch_p if x <= epoch_p else args.lr * 0.98 ** (x - epoch_p))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def record_performance(self, train_loader, val_loader, fold_i, epoch, metrics_statistics):
        save_dir = "{}/{}/{:0>4}".format(self.record_dir, fold_i, epoch)
        torch.save(self.model.state_dict(), '{}/fold-{}_{:0>4}.pth'.format(self.cpt_dir, fold_i, epoch))

        metrics_statistics.metric_values["learning rate"].append(self.optimizer.param_groups[0]['lr'])

        def record_dataloader(dataloader, loader_type="val", is_complete=True):
            for images, prompt_points, prompt_type, selected_components, sample_ids in dataloader:
                images, labels, prompt_type = map(to_cuda, (images, selected_components, prompt_type))
                images, original_size, prompt_points = self.make_prompts(images, prompt_points)
                preds = self.model(images, original_size, prompt_points, prompt_type)
                metrics_statistics.metric_values["loss_"+loader_type].append(self.loss_func(preds, labels).cpu().item())

                if is_complete:
                    preds = torch.gt(preds, 0.8).int()
                    sample_id = str(sample_ids[0])

                    image, label, pred = map(lambda x:x[0][0].cpu().detach(), (images, labels, preds))
                    prompt_points, prompt_type = prompt_points[0].cpu().detach(), prompt_type[0].cpu().detach()
                    prompt_info = np.concatenate((prompt_points, prompt_type[:,np.newaxis]), axis=1).astype(int)
                    metrics_statistics.cal_epoch_metric(
                        args.metrics, "{}-{}".format(args.label_type,loader_type), label.int(), pred.int())
                    
                    if not os.path.exists(save_dir): os.makedirs(save_dir)
                    save_sample_func = lambda x, y: np.save("/".join([save_dir,\
                                        "{}_{}_{}.npy".format(args.label_type, x, sample_id)]), y)
                    save_items = {"sample":image / 255, "label":label, "prompt_info":prompt_info, "pred":pred}
                    for x, y in save_items.items(): save_sample_func(x, y)

        record_dataloader(train_loader, "train", False)
        record_dataloader(val_loader, "val", True)

        metrics_statistics.record_result(epoch)
    
    def train(self):
        for fold_i in range(args.k_fold):
            train_loader, val_loader = self.get_dataloader(fold_i)
            self.reset()
            metrics_statistics = MetricsStatistics(save_dir="{}/{}".format(self.record_dir, fold_i))
            self.record_performance(train_loader, val_loader, fold_i, 0, metrics_statistics)
            for epoch in tqdm(range(1, args.epochs+1), desc="training"):
                for images, prompt_points, prompt_type, selected_components, sample_ids in train_loader:
                    images, labels, prompt_type = map(to_cuda, (images, selected_components, prompt_type))
                    images, original_size, prompt_points = self.make_prompts(images, prompt_points)
                    self.optimizer.zero_grad()
                    preds = self.model(images, original_size, prompt_points, prompt_type)
                    self.loss_func(preds, labels).backward()
                    self.optimizer.step()
                self.scheduler.step()
                if epoch % args.check_interval == 0: 
                    self.record_performance(train_loader, val_loader, fold_i, epoch, metrics_statistics)
            metrics_statistics.close()
        
    def make_prompts(self, images, prompt_points):
        original_size = tuple(images.shape[-2:])
        images = self.sam_transform.apply_image_torch(images)
        prompt_points = self.sam_transform.apply_coords_torch(prompt_points, original_size)

        return images, original_size, prompt_points

if __name__=="__main__":
    ppn, pnn = args.prompt_positive_num, args.prompt_negative_num
    dataset_params = [args.fov, args.label_type, ppn, pnn, args.is_local, True]
    dataset_train = octa500_2d_dataset(*dataset_params)
    dataset_params[-1] = False
    dataset_val = octa500_2d_dataset(*dataset_params)
    train_manager = TrainManager_OCTA(dataset_train, dataset_val)
    train_manager.train()