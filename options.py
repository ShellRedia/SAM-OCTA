import argparse

parser = argparse.ArgumentParser(description='training argument values')

def add_training_parser(parser):
    parser.add_argument("-device", type=str, default="0,1,2,3", help="device")
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-lr", type=float, default=5e-3)
    parser.add_argument("-check_interval", type=int, default=10)
    parser.add_argument("-k_fold", type=str, default=10)
    parser.add_argument("-prompt_positive_num", type=int, default=-1)
    parser.add_argument("-prompt_total_num", type=int, default=-1)
    parser.add_argument("-model_type", type=str, default="vit_h")
    parser.add_argument("-is_local", type=bool, default=True)
    parser.add_argument("-remark", type=str, default="OCTA-500")

def add_cell_parser(parser):
    parser.add_argument("-metrics", type=list, default=["Dice", "Jaccard", "Hausdorff"])

def add_octa500_2d_parser(parser):
    parser.add_argument("-fov", type=str, default="3M")
    parser.add_argument("-label_type", type=str, default="LargeVessel") # "Artery", "Vein"
    parser.add_argument("-metrics", type=list, default=["Dice", "Jaccard", "Hausdorff"])