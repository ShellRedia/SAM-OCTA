import argparse

parser = argparse.ArgumentParser(description='training argument values')

def add_training_parser(parser):
    parser.add_argument("-device", type=str, default="6,7", help="device")
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-batch_size", type=int, default=8)
    parser.add_argument("-lr", type=float, default=1e-3)
    parser.add_argument("-check_interval", type=int, default=5)
    parser.add_argument("-data_dir", type=str, default="datasets/OCTA-500")
    parser.add_argument("-k_fold", type=str, default=10)
    parser.add_argument("-remark", type=str, default="SAM-LoRA")

def add_octa500_2d_parser(parser):
    parser.add_argument("-fovs", type=list, default=["3M", "6M"])
    parser.add_argument("-modals", type=list, default=["OCTA"]) # "OCTA", "OCT", 
    parser.add_argument("-projection_layers", type=list, default=["OPL_BM", "ILM_OPL", "FULL"])
    parser.add_argument("-label_types", type=list, default=["LargeVessel"]) # "Artery", "Vein", "Keypoints"
    parser.add_argument("-metrics", type=list, default=["Dice", "Jaccard", "Hausdorff"])