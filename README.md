# SAM-OCTA

中文版README: [README_zh](./README_zh.md)

## 1.Quick Start

This project involves fine-tuning SAM using LoRA and performing segmentation tasks on OCTA images, built with **PyTorch**.

First, you should put a pertrained weight file in the **sam_weights** folder. The download link for pre-trained weights is as follows:

vit_h (default): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth 

vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

After testing, the required RAM for the three models is as follows: **36,248 MB, 26,154 MB, 13,467 MB**. The "vit_h" is the default option. If you need to use other smaller models, please download the corresponding weights and modify the configuration in **options.py**.

    ...
    parser.add_argument("-model_type", type=str, default="vit_h")
    ...

Use **train_sam_octa.py** to start fine-tuning. The warning informations will tell you which packages you should install. These packages are commonly used Python libraries without additional configuration.

    python train_sam_octa.py

The dataset should be formed as **OCTA-500**, like this:

    /datasets
        /OCTA-500
            /OCTA_3M
                /GT_Artery
                    10301.bmp
                    10302.bmp
                    ...
                /GT_Capillary
                    10301.bmp
                    10302.bmp
                    ...
                /GT_FAZ
                ...
                /ProjectionMaps
                    /OCTA(FULL)
                        10301.bmp
                        10302.bmp
                        ...
                    /OCTA(ILM_OPL)
                        10301.bmp
                        10302.bmp
                        ...
                    /OCTA(OPL_BM)
                        10301.bmp
                        10302.bmp
                        ...
            /OCTA_6M
                ...

Here, I used the sample with ID 10301 from the **OCTA_500** dataset of 3M FoV (Field of View) as an example. If you need the complete dataset, please contact the author of the **OCTA_500** dataset.

**OCTA-500**'s related paper: https://arxiv.org/abs/2012.07261

The results and metrics will recorded in the **results** folder (If it doesn't exist, it will be created).

If you need to visualize the prediction samples of results, please use the **display.py** file. Since the result folders are generated based on time, you may need to replace this line of code. The generated images are in the **sample_display** folder.

    ..
        test_dir = "results/2024-01-01-08-17-09/3M_LargeVessel_100_True/0/0000" # Your result dir
    ...

Here are some segmentation samples with prompt points, respectively the input image, the ground-truth and the prediction from left to right.

**Local Model**

*Artery*

![Sample](./figures/sample_artery.gif)

*FAZ*

![Sample](./figures/sample_FAZ.gif)

**Global Model**

*RV*

![Sample](./figures/sample_RV.gif)

*Capillary*

![Sample](./figures/sample_capillary.gif)

## 2.Configuration

The project can support multiple segmentation tasks and it has two modes: **global** and **local**. In fact, the performance in the global mode is comparable to other segmentation models, while the local mode is unique to SAM-OCTA. In the **options.py** file, you can configure it, and below are explanations for each option:

* -device: Specifies the IDs of available GPUs. It can support multiple GPUs, but due to the SAM code implementation by Meta, the batch_size should be equal to the number of GPUs used. However, the dataloader need to align the batch size, it is preferable to train with **batch_size=1** to avoid the prompt points mistake of different length.
* -epochs: Specifies the number of training epochs.
* -lr: The maximum learning rate, considering the warm-up strategy.
* -check_interval: Specifies how often to save results (including weights) after a certain number of training epochs.
* -k_fold: Specifies the number of folds for k-fold cross-validation.
* -prompt_positive_num: Number of positive prompt points, -1 for random.
* -prompt_total_num: Total number of prompt points, -1 for random.
* -model_type: Selects the SAM model for fine-tuning: "vit_h", "vit_l", and "vit_b".
* -is_local: Specifies whether it is in local mode.
* -remark: Some remarks you need to fill in, which will be added to the generated result folder name.

The following are some configurations specific to the OCTA-500 dataset:

* -fov: Selects the sub-dataset corresponding to the field of view.
* -label_type: Selects the annotation type (segmentation task type): "LargeVessel", "FAZ", "Capillary", "Artery", and "Vein".
* -metrics: Selects the metrics to be computed (can select multiple): "Dice", "Jaccard", "Hausdorff".

## 3.Others

 If you find the information useful, please cite the relevant paper:  https://arxiv.org/abs/2309.11758

**Pretrained Weights (Baidu Cloud Storage)**: 

Link：https://pan.baidu.com/s/1S43QadZlhT8dL8TPbA0N6g?pwd=sifh 

Password：sifh 

## 4.Instance Prediction (Supplement)
Here, I provide additional code for vessel prediction, along with explanations through text and images.

1. Prepare an Image for Prediction. Start by preparing an image that you want to predict. In the provided code, I process the image by stacking its three channels and then duplicating it side-by-side. The duplicated version is used for manual annotation of prompt points (pure green for positive points, pure red for negative points). It looks something like this:

![Sample](./figures/sample_3ch.png)


2. Load the Pretrained Weights. Since this is just an example, I use a fine-tuned ViT-L model, which requires less memory and computation time. The provided weights combine both global and local prediction modes. You can download the weights from the following link:

https://pan.baidu.com/s/1iCVmPaLOWVk36YbgcQ4AOg?pwd=i54c password: i54c 

Then, run the script predict.py, and the results will be saved in an automatically generated folder named prediction.

3. In global mode, no prompt points are needed. I automatically added a fixed negative point at [-100, -100] in the code. Let's take a look at the segmentation results:

![Sample](./figures/pred_rv_global.png)

4. In local mode, prompt points are provided on the vessels, for example:

![Sample](./figures/sample_3ch_prompt.png)
![Sample](./figures/sample_3ch_prompt2.png)

The results for the provided prompt points are as follows:

![Sample](./figures/pred_rv_local.png)
![Sample](./figures/pred_rv_local2.png)


