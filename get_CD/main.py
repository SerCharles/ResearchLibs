from dataloader import load_datas
from chamfer_layer import ChamferDistKDTree
import argparse
import torch

def init_args():
    parser = argparse.ArgumentParser(description = "Get CD")
    parser.add_argument("--category", type = str, default = None, help = "The name of categories")
    parser.add_argument("--run_name", type = str, default = 'all', help = "The name of run")
    parser.add_argument("--nsamples", type = int, default = 512, help = "The name of points")
    parser.add_argument("--use_norm", default = 0, type = int)
    parser.add_argument("--normals", default = False, type = bool)
    parser.add_argument("--loss", default = "l2", type = str)
    args = parser.parse_args()
    if args.use_norm == 0:
        args.normals = False
    elif args.use_norm == 1:
        args.normals = True
    if args.category == None:
        args.run_name == 'all'
    else: 
        args.run_name = args.category
    return args

if __name__ == "__main__":
    losses = {"l1" : torch.nn.L1Loss(), 'l2' : torch.nn.MSELoss(), 'huber' : torch.nn.SmoothL1Loss()}
    args = init_args()
    print(args)
    data_list = load_datas(args)
    print("Data loaded")
    chamfer_layer = ChamferDistKDTree()
    criterion = losses[args.loss]

    total_loss = 0.0
    for pair in data_list:
        source, target = pair
        source = source.reshape(1, source.shape[0], source.shape[1])
        target = target.reshape(1, target.shape[0], target.shape[1])
        source = torch.from_numpy(source)
        target = torch.from_numpy(target)
        _, _, dist = chamfer_layer(source, target)
        loss = criterion(dist, torch.zeros_like(dist))
        total_loss += loss.item()
        print(loss.item())
    total_loss /= len(data_list)
    print("Average CD loss is {:.6f}".format(total_loss))