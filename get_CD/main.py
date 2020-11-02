from dataloader import load_datas
from chamfer_layer import ChamferDistKDTree
import argparse

def init_args():
    parser = argparse.ArgumentParser(description = "Get CD")
    parser.add_argument("--category", type = str, default = None, help = "The name of categories")
    parser.add_argument("--run_name", type = str, default = 'all', help = "The name of run")
    parser.add_argument("--nsamples", type = int, default = 512, help = "The name of points")
    parser.add_argument("--use_norm", default = 0, type = int)
    parser.add_argument("--normals", default = False, type = bool)
    args = parser.parse_args()
    if args.use_norm == 0:
        args.normals = False
    elif args.use_norm == 1:
        args.normals = True
    return args

if __name__ == "__main__":
    args = init_args()
    data_list = load_datas(args)
    chamfer_layer = ChamferDistKDTree()
    for pair in data_list:
        source, target = pair
        _, _, dist = chamfer_layer(source, target)