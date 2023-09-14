# SAM-OCTA

This project involves fine-tuning SAM using LoRA and performing segmentation tasks on OCTA images.

This project is built using PyTorch and requires a GPU with 24GB or more of RAM.

Use **train.py** to start fine-tuning. The warning informations will tell you which packages you should install.

The dataset should be formed as **OCTA-500**, like:

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

The results and metrics will recorded in the **results** folder (If it doesn't exist, it will be created).

Here is a sample of local vein segmentation with prompt points.

![Sample](./figures/sample.png)